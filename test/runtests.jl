using VectorizedRNG
using Test

using RNGTest, Distributions, Random

@testset "VectorizedRNG.jl" begin
    # Write your own tests here.


    struct RandNormal01{T<:VectorizedRNG.AbstractPCG} <: Random.AbstractRNG
        pcg::T
    end
    function Random.rand!(r::RandNormal01, x::AbstractArray)
        randn!(r.pcg, x)
        x .= cdf.(Normal(0,1), x)
    end
    rn1 = RandNormal01(local_pcg());
    rng = RNGTest.wrap(rn1, Float64);
    RNGTest.smallcrushTestU01(rng)
    RNGTest.bigcrushTestU01(rng)
    
end
