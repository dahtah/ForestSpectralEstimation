import Polynomials as Poly

#Some code for handling (classical) moments
#In this file a moment sequence is represented by a vector
#s = [s[0],s[1],...,s[k-1]]
#where s[i] = E(x^i)
#Note that generally s[0] = 1

#Classical moment matrix
function moment_matrix(s)
    m = length(s)-1
    n = iseven(m) ? div(m,2) : div(m-1,2)
    s = OffsetArrays.Origin(0)(s)
    [s[i + j] for i in 0:n,j in 0:n]
end


#Given a moment sequence, find upper and lower bounds for the next moment
#Uses Hankel determinants; see e.g. Dette & Studden, Theory of Canonical Moments
#p. 20
function conditional_moment_bounds(s,a=-1.0,b=1.0)
    m = length(s)-1
    s = OffsetArrays.Origin(0)(s)
    if m == 0
        return a,b
    end
    if m == 1
        up=a+b-a*b
        lw=s[1]^2
        return lw,up
    end
    if isodd(m)
        n = div(m-1,2)
        Hl=[s[i + j] for i in 0:n,j in 0:n]
        z=[s[1+n+i] for i in 0:n]


        lw=z'*(Symmetric(Hl)\z)
        Hu=[(a+b)*s[i+j+1] - s[i+j+2] - a*b*s[i+j] for i in 0:(n-1),j in 0:(n-1)]
        z=[(a+b)*s[i+n+1] - s[i+n+2] - a*b*s[i+n] for i in 0:(n-1)]
        up= - z'*(Hu\z) + (a+b)*s[2n+1]-a*b*s[2n]
        lw,up
    else
        n = div(m-2,2)
        Hl=[s[i + j + 1] - a* s[i+j] for i in 0:n,j in 0:n]
        z=[s[n+1 + j + 1] - a* s[n+1+j] for j in 0:n]
        lw=z'*(Hl\z)+a*s[end]
        Hu=[b*s[i + j] - s[i+j+1] for i in 0:n,j in 0:n]
        z=[b*s[n+1 + j] - s[n+1+j+1] for j in 0:n]
        up=-z'*(Hu\z)+b*s[end]
        lw,up
    end
end

function canonical_moments(s,a=-1.0,b=1.0)
    c = zeros(length(s))
    c[1] = s[1]
    for i in 2:length(s)
        l,u = conditional_moment_bounds(s[1:(i-1)],a,b)
        c[i] = (s[i]-l) / (u-l)
    end
    c
end

#Various matrices used to ensure that the moment sequence s is admissible.
#see e.g. Dette & Studden, Theory of Canonical Moments p. 20
function Hmat(s,a=-1,b=1)
    m = length(s)-1
    s = OffsetArrays.Origin(0)(s)
    if iseven(m)
        n = div(m,2)
        Hl=[s[i + j] for i in 0:n,j in 0:n]
        Hu=[(a+b)*s[i+j+1] - s[i+j+2] - a*b*s[i+j] for i in 0:(n-1),j in 0:(n-1)]
        Hl,Hu
    else
        n = div(m-1,2)
        Hl=[s[i + j + 1] - a* s[i+j] for i in 0:n,j in 0:n]
        Hu=[b*s[i + j] - s[i+j+1] for i in 0:n,j in 0:n]
        Hl,Hu
    end
end

#Verify that moment sequence s is admissible on the interval [a,b]
#ie that there exists a measure with support on [a,b] that has s has its (truncated) moment vector
#see Schmüdgen, The Moment Problem, p. 230
function is_admissible(s,a=-1.0,b=1.0)
    all(map(isposdef,Hmat(s,a,b)))
end

#find largest degree k such that s[1:k] is admissible
#stupid implementation, use only for low degrees!
function admissible_subset(s,a=-1.0,b=1.0)
    k = length(s)
    while k > 0
        if is_admissible(s[1:k],a,b)
            break
        else
            k -= 1
        end
    end
   return s[1:k]
end

#Given a moment sequence under a measure μ, transform to a moment sequence
#for a measure with weight (x-a) dμ (type == :a)
#(b-x) dμ (type == :b)
#(x-a)(b-x) dμ (type == :ab)
function transformed_sequence(s,type=:a,a=-1,b=1)
    if type == :a
        [s[i]-a*s[i-1] for i in 2:length(s)]
    elseif type == :b
        [b*s[i-1]-s[i] for i in 2:length(s)]
    elseif type == :ab
        [-s[i]+(a+b)*s[i-1]-a*b*s[i-2] for i in 3:length(s)]
    else
        error("Undefined transform type")
    end
end


#Theorem 5.5 in Golub&Meurant (2010), Matrices, moments and quadrature p. 58
function jacobi_from_moments(s)
    if length(s) == 2
        return (α=[s[2]/s[1]],η=Float64[])
    else
        #H = Hmat(s)[1]
        H = moment_matrix(s)
        k = size(H,1)-1
        R = cholesky(H).L'
        η = [R[j+1,j+1]/R[j,j] for j in 1:(k-1)]
        α = zeros(k)
        α[1] = R[1,2]
        for j in 2:(k)
            α[j] = R[j,j+1]/R[j,j] - R[j-1,j]/R[j-1,j-1]
        end
        return (α=α,η=η)
    end
end

function gaussquad_from_moments(s)
    λ,U=eigen(SymTridiagonal(jacobi_from_moments(s)...))
    λ,U[1,:].^2
end

function vdm(x,deg=length(x)-1)
    [x^i for x in x, i in 0:deg]
end

function lower_representation(s,a=-1.,b=1.)
    n = length(s)
    if iseven(n)
        x=roots_from_moments(s)
        w = (vdm(x)') \ s[1:(length(x))]
        (x,w)
    else
        sp = transformed_sequence(s,:a,a,b)
        #compute inner nodes
        x=roots_from_moments(sp)
        #solve VdM system
        w = (vdm([a;x])') \ s[1:(length(x)+1)]
        ([a;x],w)
    end
end

function upper_representation(s,a=-1.,b=1.)
    n = length(s)
    if iseven(n)
        sp = transformed_sequence(s,:ab,a,b)
        #compute inner nodes
        x = roots_from_moments(sp)
        #solve VdM system
        w = (vdm([a;x;b])') \ s[1:(length(x)+2)]
        ([a;x;b],w)
    else
        sp = transformed_sequence(s,:b,a,b)
        #compute inner nodes
        x = roots_from_moments(sp)
        #solve VdM system
        w = (vdm([x;b])') \ s[1:(length(x)+1)]
        ([x;b],w)
    end
end

function kernel_roots(C,α,ξ)
    Vs = [Poly.Polynomial(C[:,i])/sqrt(α[i]) for i in 1:size(C,1)]
    Vv = [v(ξ) for v in Vs]
    Q = sum([Vs[i]*Vv[i] for i in 1:length(Vv)])
    Poly.roots(Q)
    #Q
end

#Compute moments of a quadrature rule 
function moments_qr(x,w,deg)
    [dot(w,x.^j) for j in 0:deg]
end

#Find quadrature weights given a set of nodes
function fit_quadweights(x,s)
    V=vdm(x,length(s)-1)
    w = V' \ s
    valid = all(w .>= 0) && (V'*w ≈ s)
    return (w=w,valid=valid)
end

#Markov bound for μ([a,ξ]) based on moment sequence s
function markov_bound(s,ξ,a=-1.,b=1.)
    if ξ == a
        return (0.0,0.0)
    elseif ξ == b
        return (1.0,1.0)
    else
        x,w=canonical_representation(s,ξ,a,b)
        lw = sum(w[x .< ξ])
        up = sum(w[x .<= ξ])
        (lw,up)
    end
end

function canonical_representation(s,ξ,a=-1.,b=1.)
    n = length(s)
    if isodd(n)
        sp = transformed_sequence(s,:ab,a,b)
        C,α=monic_orth_poly(sp)
        x=[a;kernel_roots(C,α,ξ);ξ;b]
        w,valid=fit_quadweights(x,s)
        if (valid)

            return x,w
        else
            C,α=monic_orth_poly(s)
            x=[kernel_roots(C,α,ξ);ξ]
            w,valid=fit_quadweights(x,s)
            @assert valid
            return x,w
        end
    else
        sp = transformed_sequence(s,:a,a,b)
        C,α=monic_orth_poly(sp)
        x=[a;kernel_roots(C,α,ξ);ξ]
        w,valid=fit_quadweights(x,s)
        if (valid) #Found a valid solution
            return x,w
        else
            sp = transformed_sequence(s,:b,a,b)
            C,α=monic_orth_poly(sp)
            x=[kernel_roots(C,α,ξ);b;ξ]
            w,valid=fit_quadweights(x,s)
            @assert valid
            return x,w
        end
    end
end



function mb_rep(s,a=-1,b=1)
    n = length(s)
    qd = [lower_representation(s,a,b),upper_representation(s,a,b)]
    [dot(q[2],q[1].^(n)) for q in qd]
end

#Computes the coefficients of the normalised orthogonal polynomials in the
#monomial basis Warning: very poor numerical stability!
function orth_from_moments(s)
    H = moment_matrix(s)
    R = cholesky(H).L
    inv(R)
end

function shift_op(s)
    sc = copy(s)
    for i in eachindex(s)[1:(end-1)]
        sc[i+1]=s[i]
    end
    sc
end


#Run the (Arnoldi-)Gram-Schmidt algorithm to generate the coefficients
#for the monic orthogonal polynomials in terms of the monomials
#Returns a tuple (C,α) where C are the coefficients and α is a vector containing the norms
function monic_orth_poly(s)
    l = length(s)-1
    if iseven(l)
        r = div(l,2)
        dmax = r
    else
        r = div(l-1,2)
        dmax = r+1
        s = [s;0.0] #pad
    end
    s = OffsetArrays.Origin(0)(s)
    C=OffsetArrays.Origin(0)(zeros(dmax+1,dmax+1))
    C[0,0] = 1
    α=OffsetArrays.Origin(0)(zeros(dmax+1)) #Squared norms 
    α[0] = s[0]
    dp = (u,v,deg) -> sum((u[i]*v[j]*s[i+j] for i = 0:(deg),j = 0:(deg)))
    for deg = 1:dmax
        if (deg == 1)
            c = C[:,deg-1]
            sc = circshift(c,1)
            β=dp(c,sc,1)/ (α[deg-1])
            C[:,deg] = sc - β*c
        else
            cm=C[:,deg-1]
            cmm=C[:,deg-2]
            sm=circshift(cm,1)
            #smm=shift_op(cmm)
            α[deg-1] = dp(cm,cm,deg-1)
            β=dp(sm,cm,deg)/α[deg-1]
            γ=dp(sm,cmm,deg)/α[deg-2]
            C[:,deg] = sm - β*cm - γ*cmm
        end
    end
    if iseven(l)
        α[dmax] = dp(C[:,dmax],C[:,dmax],dmax)
    end
    (R=OffsetArrays.no_offset_view(C),α=OffsetArrays.no_offset_view(α))
end

#Roots of the last monic orth polynomial computable from moment sequence s
function roots_from_moments(s)
    R,_ = monic_orth_poly(s)
    Poly.roots(Poly.Polynomial(R[:,end]))
end

#Given the moments s of random variable x, find the moments
#of α*(x+β)
function rescale_moment_sequence(s,α,β)
    deg = length(s)-1
    s = OffsetArrays.Origin(0)(s)
    st = copy(s)
    for i in 0:deg
        v = 0.0
        for j in 0:i
            v+= binomial(i,j)*s[j]*β^(i-j)
        end
        st[i] = (α^i ) * v
    end
    OffsetArrays.no_offset_view(st)
end

#Moments of the uniform distribution on [-1,1]
function legendre_moments(deg)
    [iseven(i)/(i+1) for i in 0:deg]
end

#Moments of the uniform distribution on [a,b]
function uniform_moments(deg,a=-1.,b=1.)
    s = legendre_moments(deg)
    r = (b-a)/2
    rescale_moment_sequence(s,r,1+2a/(b-a))
end
