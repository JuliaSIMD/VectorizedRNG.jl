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

const INVSQRT2 = 1/sqrt(2)
@inline function normalcdf(v)
    T = eltype(v)
    T(0.5) * ( one(T) + VectorizedRNG.SIMDPirates.verf( v * INVSQRT2 ) )
end
function normalcdf!(x::AbstractVector{T}) where {T}
    _W, Wshift = VectorizedRNG.VectorizationBase.pick_vector_width_shift(T)
    W = VectorizedRNG.VectorizationBase.pick_vector_width_val(T)
    N = length(x)
    ptrx = pointer(x)
    i = 0
    for _ ∈ 1:(N >>> Wshift)
        ptrxᵢ = VectorizedRNG.VectorizationBase.gep(ptrx, i)
        v = VectorizedRNG.SIMDPirates.load(W, ptrxᵢ)
        VectorizedRNG.SIMDPirates.store!(ptrxᵢ, normalcdf(v))
        i += _W
    end
    if i < N
        ptrxᵢ = VectorizedRNG.VectorizationBase.gep(ptrx, i)
        mask = VectorizedRNG.VectorizationBase.mask(T, N & (_W - 1))
        v = VectorizedRNG.SIMDPirates.load(W, ptrxᵢ, mask)
        VectorizedRNG.SIMDPirates.store!(ptrxᵢ, normalcdf(v), mask)
    end
    x
end
    struct RandNormal01{T<:VectorizedRNG.AbstractPCG} <: Random.AbstractRNG
        pcg::T
    end
    function Random.rand!(r::RandNormal01, x::AbstractArray)
        randn!(r.pcg, x)
        normalcdf!(x)
        # x .= cdf.(Normal(0,1), x)
    end
    rngnorm = RNGTest.wrap(RandNormal01(local_pcg()), Float64);


@testset "VectorizedRNG.jl" begin
    # Write your own tests here.

    
    rngunif = RNGTest.wrap(local_pcg(), Float64);
    res = RNGTest.smallcrushJulia(rngunif)
    @test smallcrushminp(res) > eps()

    rngnorm = RNGTest.wrap(RandNormal01(local_pcg()), Float64);
    res = RNGTest.smallcrushJulia(rngnorm)
    @test smallcrushminp(res) > eps()
    
end


