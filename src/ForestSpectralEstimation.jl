module ForestSpectralEstimation
using FastGaussQuadrature,Statistics,Distributions,LegendrePolynomials,LogExpFunctions,LinearAlgebra,SparseArrays,Optim,Graphs,KirchhoffForests,QuadGK
import LineSearches,OffsetArrays,StatsBase

include("exp_family.jl")
include("jackson_cheb.jl")
include("chebmoments.jl")
include("slq.jl")

include("moments.jl")
include("markov_bounds.jl")
include("denoise.jl")
include("fixed_q_estimation.jl")
include("isotonic.jl")
include("global.jl")
end # module ForestSpectralEstimation
