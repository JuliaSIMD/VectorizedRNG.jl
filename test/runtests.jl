using VectorizedRNG
using Test

using RNGTest, Random#, Distributions

const α = 1e-4

function smallcrushextrema(res)
    r1 = Base.Cartesian.@ntuple 5 i -> (res[i])::Float64
    r2 = res[6]::Tuple{Float64,Float64}
    r3 = Base.Cartesian.@ntuple 3 i -> (res[i+6])::Float64
    r4 = res[10]::NTuple{5,Float64}
    mi = min(
        minimum(r1), minimum(r2), minimum(r3), minimum(r4)
    )
    ma = max(
        maximum(r1), maximum(r2), maximum(r3), maximum(r4)
    )
    mi, ma
end

const INVSQRT2 = Float64(1/sqrt(big(2)))
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
        v = VectorizedRNG.SIMDPirates.vload(W, ptrxᵢ)
        VectorizedRNG.SIMDPirates.vstore!(ptrxᵢ, normalcdf(v))
        i += _W
    end
    if i < N
        ptrxᵢ = VectorizedRNG.VectorizationBase.gep(ptrx, i)
        mask = VectorizedRNG.VectorizationBase.mask(T, N & (_W - 1))
        v = VectorizedRNG.SIMDPirates.vload(W, ptrxᵢ, mask)
        VectorizedRNG.SIMDPirates.vstore!(ptrxᵢ, normalcdf(v), mask)
    end
    x
end
    struct RandNormal01{T<:VectorizedRNG.AbstractVRNG} <: Random.AbstractRNG
        rng::T
    end
    function Random.rand!(r::RandNormal01, x::AbstractArray)
        randn!(r.rng, x)
        normalcdf!(x)
        # x .= cdf.(Normal(0,1), x)
    end
    # rngnorm = RNGTest.wrap(RandNormal01(local_pcg()), Float64);



@testset "VectorizedRNG.jl" begin
    # Write your own tests here.

    rngunif = RNGTest.wrap(local_rng(), Float64);
    res = RNGTest.smallcrushJulia(rngunif)
    mi, ma = smallcrushextrema(res)
    @test mi > α
    @test ma < 1 - α

    # rngunif = RNGTest.wrap(local_pcg(), Float64);
    # res = RNGTest.smallcrushJulia(rngunif)
    # mi, ma = smallcrushextrema(res)
    # @test mi > α
    # @test ma < 1 - α

    rngnorm = RNGTest.wrap(RandNormal01(local_rng()), Float64);
    res = RNGTest.smallcrushJulia(rngnorm)
    mi, ma = smallcrushextrema(res)
    @test mi > α
    @test ma < 1 - α

    # rngnorm = RNGTest.wrap(RandNormal01(local_pcg()), Float64);
    # res = RNGTest.smallcrushJulia(rngnorm)
    # mi, ma = smallcrushextrema(res)
    # @test mi > α
    # @test ma < 1 - α

end


