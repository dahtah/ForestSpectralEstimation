module ForestSpectralEstimation
using FastGaussQuadrature,Statistics,Distributions,LegendrePolynomials,LogExpFunctions,LinearAlgebra,SparseArrays,Optim,Graphs,KirchhoffForests
import LineSearches,OffsetArrays

include("exp_family.jl")
include("jackson_cheb.jl")

include("moments.jl")
include("markov_bounds.jl")
include("denoise.jl")
include("fixed_q_estimation.jl")
include("isotonic.jl")
end # module ForestSpectralEstimation
