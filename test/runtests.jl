import ForestSpectralEstimation as FSE
using Test
using LinearAlgebra,Statistics

const testdir = dirname(@__FILE__)
tests = ["moments"]
@testset "ForestSpectralEstimation" begin
    for t in tests
        tp = joinpath(testdir, "$(t).jl")
        include(tp)
    end
end

