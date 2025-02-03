#Some code related to max. entropy estimation
#This sets up an exponential family on the interval [-1,1]

#FIXME: refactor to support arbitrary domains

#Exponential family wrt to base measure defined by Jacobi density
#(1-x)^α (1+x)^β on [-1,1] interval
struct ExpFamily{T}
    x :: Vector{Float64}
    w :: Vector{Float64}
    S :: Matrix{Float64}
    α :: Float64
    β :: Float64
    sfun :: T
end

function Base.show(io::IO, ef::ExpFamily)
    println(io, "Exponential family with $(nmoments(ef)) moments.")
end

function ExpFamily(sfun,nq=200;α=0.0,β=0.0,τ=0.0)
    quad = gaussjacobi(nq,α,β)
    x = quad[1]
    w = quad[2]/sum(quad[2])
    S = reduce(hcat,map(sfun,x))
    ExpFamily{typeof(sfun)}(x,w,S',α,β,sfun)
end

function nquad(ef::ExpFamily)
    length(ef.x)
end

function nmoments(ef::ExpFamily)
    size(ef.S,2)
end

#base measure as a rescaled Beta density
function distr_base(ef :: ExpFamily)
    2Beta(ef.β+1,ef.α+1) - 1
end

function base_pdf(ef :: ExpFamily,x)
    pdf(distr_base(ef),x)
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
    err = norm(compute_moments(ef,βs(v))[ind]-μ[ind])
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
    xr = range(-1,1,1000)
    δ = xr[2]-xr[1]
    δ*sum((pdf(ed,v) for v in xr if v <= x))
end
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
function maxent_confidence(ef,μ,std_err;η0 = zeros(length(μ)),step_size=1.0,maxiter=100,max_ls=30)
    m = length(μ)
    η = η0
    ind_iter=0

    cfun = (v)->logz(ef,v)-dot(v,μ)

    while ind_iter < maxiter
        ed=ExpFamilyDistribution(ef;β=η)
        res=(μ-ed.μ) ./ std_err
#        @show norm(res),cfun(η)
#        ed=ExpFamilyDistribution(ef,β=zeros(2))
        if norm(res) < 1
            break
        end
        g = ed.μ-μ
        C=cholesky(Symmetric(cov_ss(ed)))
        α = step_size
        δ = C\g
        c0 = cfun(η)
        ind_ls = 0
        while (cfun(η-α*δ) > c0) && (ind_ls < max_ls)
 #           @show α
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
