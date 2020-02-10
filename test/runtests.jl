using VectorizedRNG
using Test

using RNGTest, Distributions, Random

function smallcrushminp(res)
    r1 = Base.Cartesian.@ntuple 5 i -> (res[i])::Float64
    r2 = res[6]::Tuple{Float64,Float64}
    r3 = Base.Cartesian.@ntuple 3 i -> (res[i+6])::Float64
    r4 = res[10]::NTuple{5,Float64}
    min(
        minimum(r1), minimum(r2), minimum(r3), minimum(r4)
    )
end

@testset "VectorizedRNG.jl" begin
    # Write your own tests here.

    
    rngunif = RNGTest.wrap(local_pcg(), Float64);
    res = RNGTest.smallcrushJulia(rngunif)
    @test smallcrushminp(res) > eps()

    struct RandNormal01{T<:VectorizedRNG.AbstractPCG} <: Random.AbstractRNG
        pcg::T
    end
    function Random.rand!(r::RandNormal01, x::AbstractArray)
        randn!(r.pcg, x)
        x .= cdf.(Normal(0,1), x)
    end
    rngnorm = RNGTest.wrap(RandNormal01(local_pcg()), Float64);
    res = RNGTest.smallcrushJulia(rngunif)
    @test smallcrushminp(res) > eps()
    
end
