#Some code related to max. entropy estimation
#This sets up an exponential family on the interval [-1,1]

#FIXME: refactor to support arbitrary domains

#Exponential family wrt to base measure defined
#by uniform density on [a,b]
struct ExpFamily{T}
    x :: Vector{Float64}
    w :: Vector{Float64}
    S :: Matrix{Float64}
    a :: Float64
    b :: Float64
    sfun :: T
end

function Base.show(io::IO, ef::ExpFamily)
    println(io, "Exponential family with $(nmoments(ef)) moments.")
end

#map from standard interval [-1,1] to [a,b]
function inv_map_si(x,a,b)
    .5*(b-a)*(x+1) + a
end

#map from [a,b] to [-1,1] 
function map_si(x,a,b)
    2*(x-a)/(b-a) - 1
end

function ExpFamily(sfun;a=-1.0,b=1.0,τ=0.0,nq=200)
    #    quad = gaussjacobi(nq,α,β)
    quad = gausslegendre(nq)
    x = inv_map_si.(quad[1],a,b)
    w = quad[2]/sum(quad[2])
    S = reduce(hcat,map(sfun,x))
    ExpFamily{typeof(sfun)}(x,w,S',a,b,sfun)
end

function nquad(ef::ExpFamily)
    length(ef.x)
end

function nmoments(ef::ExpFamily)
    size(ef.S,2)
end


function distr_base(ef :: ExpFamily)
    Uniform(ef.a,ef.b)
end

function base_pdf(ef :: ExpFamily,x)
    (ef.a <= x <= ef.b ? 1/(ef.b-ef.a) : 0.0)
end

function base_mean(ef :: ExpFamily)
    mean(distr_base(ef))
end

function base_logpdf(ef :: ExpFamily,x)
    logpdf(distr_base(ef),x)
end

#Log-partition function 
function logz(ef::ExpFamily,β)
    f = ef.S*β + log.(ef.w)
    LogExpFunctions.logsumexp(f)
end

#Expectation of the summary statistics
function compute_moments(ef :: ExpFamily, β)
    vec((softmax(ef.S*β + log.(ef.w)))' * ef.S)
end

#Covariance of the summary statistics
function compute_var(ef :: ExpFamily, β)
    m = compute_moments(ef,β)
    W = Diagonal(softmax(ef.S*β + log.(ef.w)))
    (ef.S' * W * ef.S) - m*m'
end

#A (hopefully) more stable way to compute a Newton update
#This avoids taking the Cholesky decomposition of the covariance matrix
#which can be very ill-conditioned
#Instead, we run QR on basis functions rescaled by density
function compute_newton_update(ef :: ExpFamily, β,μ)
    m = compute_moments(ef,β)
    W = Diagonal(softmax(ef.S*β + log.(ef.w)))
    R=qr(sqrt(W)*ef.S).R
    c = R \ (R' \ m)
    b= R \ (R' \ (m-μ)) +  (c*dot(c,m-μ))/(1-dot(m,c))
end

function compute_entropy(ef::ExpFamily,β)
    f = ef.S*β
    g = exp.(f)
    z = dot(ef.w,g) #should probably use logsumexp
    gn = g ./ z
    fn = f .- log(z)
    - dot(ef.w, gn .* fn)
end

#Find a distribution with moments μ in the exp. family, with tolerance tol.
#Return a vector of natural parameters
function match_moments(ef :: ExpFamily,μ;tol=1e-4)
    cfun = (v)->logz(ef,v)-dot(v,μ)
    gfun = (v)->compute_moments(ef,v)-μ
    Hfun = (v)->compute_var(ef,v)

    res=Optim.optimize(cfun,gfun,Hfun,zeros(length(μ)),Optim.Newton(linesearch=LineSearches.BackTracking()),inplace=false)
    v=Optim.minimizer(res)
    err = norm(compute_moments(ef,v)-μ)
    if (err > tol)
        error("Could not match moments up to tolerance. Err : $(err)")
    end
    v
end

#Encodes a particular distribution within an exponential family
struct ExpFamilyDistribution
    ef :: ExpFamily
    β :: Vector{Float64}
    μ :: Vector{Float64}
    dens :: Vector{Float64}
    lz :: Float64
end

function Base.show(io::IO, ed::ExpFamilyDistribution)
    println(io, "Exponential family distribution with moments $(ed.μ).")
end


function ExpFamilyDistribution(ef::ExpFamily; β = nothing, μ = nothing)
    if μ === nothing
        @assert isa(β,Vector)
        μ=compute_moments(ef,β)
    elseif β === nothing
        @assert isa(μ,Vector)
        β=match_moments(ef,μ)
    end
    f=ef.S*β #+log.(ef.w)
    lz=LogExpFunctions.logsumexp(f + log.(ef.w) )
    ExpFamilyDistribution(ef,β,μ,exp.(f  .- lz),lz)
end

#Log PDF wrt to uniform measure on [-1,1]
function Distributions.logpdf(ed::ExpFamilyDistribution,x)
    dot(ed.ef.sfun(x),ed.β) - ed.lz + base_logpdf(ed.ef,x)
end

#PDF wrt to uniform measure on [-1,1]
function Distributions.pdf(ed::ExpFamilyDistribution,x)
    exp(logpdf(ed,x))
end


function Distributions.cdf(ed::ExpFamilyDistribution,x)
    quadgk(v->pdf(ed,v),ed.ef.a,x)[1]
end
    # function Distributions.cdf(ed::ExpFamilyDistribution,x)
#     xr = range(ed.ef.a,ed.ef.b,1000)
#     δ = xr[2]-xr[1]
#     δ*sum((pdf(ed,v) for v in xr if v <= x))
# end


function Statistics.mean(ed::ExpFamilyDistribution)
    dot(ed.dens .* ed.ef.w,ed.ef.x)
end

function Statistics.var(ed::ExpFamilyDistribution)
    m = mean(ed)
    dot(ed.dens .* ed.ef.w,(ed.ef.x .- m).^2)
end

function expectation(ed::ExpFamilyDistribution,f)
    sum( f(x)*w*d for (x,w,d) in zip(ed.ef.x,ed.ef.w,ed.dens))
end


function cov_ss(ed::ExpFamilyDistribution)
    compute_var(ed.ef,ed.β)
end

#Find a maximum entropy distribution that falls within the confidence bounds
#Uses Newton's method, stopping when we reach the confidence ball
function maxent_confidence(ef,μ,std_err;η0 = zeros(length(μ)),step_size=1.0,maxiter=100,max_ls=300)
    m = length(μ)
    η = η0
    ind_iter=0
    cfun = (v)->logz(ef,v)-dot(v,μ)
    while ind_iter < maxiter
        ed=ExpFamilyDistribution(ef;β=η)
        res=(μ-ed.μ) ./ std_err
        if norm(res) < 1
            break
        end
        α = step_size
        δ = compute_newton_update(ef,η,μ)
        c0 = cfun(η)
        ind_ls = 0
        while (cfun(η-α*δ) > c0) && (ind_ls < max_ls)
            α = .9*α
            ind_ls+=1
        end
        if ind_ls == max_ls
            error("Backtracking failed.")
        end
        η=η-α*δ
        ind_iter+=1
    end
    (ed=ExpFamilyDistribution(ef,β=η),niter=ind_iter)
end


function chebpoly(x,n,a=-1,b=1)
    cos(n*acos(map_si(x,a,b)))
end

function approx_moment_primal_cost(ef,β,m,α,C)
    h=compute_entropy(ef,β)
    r = compute_moments(ef,β) - m
    -h + (.5/(α)) * (r'*(C\r))
end

function approx_moment_chi2(ef,β,m,C)
    r = compute_moments(ef,β) - m
    (r'*(C\r))
end


function approx_moment_dual_cost(ef,β,m,α,C)
    logz(ef,β) - dot(m,β) + .5*α*β'*C*β
end


function approx_moment_inner(ef,β,m,α,C)
    cf = (v)->logz(ef,v) - dot(m,v) + .5*α*v'*C*v
    gf = (v)->compute_moments(ef,v) - m + α*C*v
    Hf= (v)->compute_var(ef,v) +α*C
    #        res=Optim.optimize(cf,gf,β,BFGS(linesearch=LineSearches.BackTracking()),inplace=false,Optim.Options(show_trace=false))
    res=Optim.optimize(cf,gf,Hf,β,Newton(linesearch=LineSearches.BackTracking()),inplace=false,Optim.Options(show_trace=false))

    res.minimizer
end
    

#Implementation of Silver et al. Efficient Maximum Entropy Algorithms for Electronic Structure 
function approx_moment_matching(ef :: ExpFamily,m,C;maxit_outer=5,tol=1)
    β=zeros(nmoments(ef))
    r = compute_moments(ef,β) - m
    α = 1/(r'*(C\r));
    it_outer=0
    @info "Residual $(1/α) at iteration $(it_outer)"
    while it_outer < maxit_outer
        β=approx_moment_inner(ef,β,m,α,C)
        r=compute_moments(ef,β) - m
        nr=r'*(C\r)
        @info "Residual $(nr) at iteration $(it_outer)"
        if (nr < tol)
            break
        end
        α=α/2
        @show α
        it_outer += 1
    end
    @info "Final residual $(r'*(C\r))"
    ExpFamilyDistribution(ef,β=β)
end
