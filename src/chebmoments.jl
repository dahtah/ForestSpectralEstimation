import Polynomials
function empirical_chebmoments(x,deg;a=-1,b=1)
    t = zeros(deg+1)
    t = OffsetArrays.Origin(0)(t)
    t[0] = 1.0
    v = map_si.(x,a,b)
    t[1] = mean(v)
    cmm = ones(length(x))
    cm = copy(v )
    for i in 2:deg
        c =2* v .* cm - cmm
        t[i] = mean(c)
        cmm .= cm
        cm .= c
    end
    OffsetArrays.no_offset_view(t)[2:end]
end

function shifted_cheb(x,deg;a=-1,b=1)
    t = acos(map_si(x,a,b))
    [cos(n*t) for n in 1:deg]
end

function cheb_basis(x,deg;a=-1,b=1)
    n=length(x)
    C = zeros(n,deg+1)
    v = map_si.(x,a,b)
    C[:,1] .= 1.0
    C[:,2] .= v
    for d in 2:deg
        C[:,d+1] = 2* v .* C[:,d] - C[:,d-1]
    end
    C
end

#Coefficients for conversion between monomial basis and Cheb basis on [a,b]
function mon2cheb(deg,a=0,b=1)
    α = 2/(b-a)
    β = -(b+a)/2
    n = deg+1
    pmm = Polynomials.Polynomial([1])
    pm = Polynomials.Polynomial([α*β,α])
    s = Polynomials.Polynomial([α*β,α])
    C = zeros(deg+1,deg+1)
    C[1,1] = 1
    C[2,1:2] .= Polynomials.coeffs(pm)
    for i in 2:deg
        p = 2*s*pm - pmm
        C[i+1,1:(i+1)] .= Polynomials.coeffs(p)
        pmm = pm
        pm =p
    end
    C
end


