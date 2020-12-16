using VectorizedRNG
using Test

using RNGTest, Random, SpecialFunctions, Aqua#, Distributions

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
# TODO: Get a new SIMD erf implementation
# @inline function normalcdf(v::VectorizedRNG.Vec{W,T}) where {W,T}
#     T(0.5) * ( one(T) + VectorizedRNG.SIMDPirates.verf( v * INVSQRT2 ) )
# end
# function normalcdf!(x::AbstractVector{T}) where {T}
#     _W, Wshift = VectorizedRNG.VectorizationBase.pick_vector_width_shift(T)
#     W = VectorizedRNG.VectorizationBase.pick_vector_width_val(T)
#     N = length(x)
#     ptrx = pointer(x)
#     i = 0
#     for _ ∈ 1:(N >>> Wshift)
#         ptrxᵢ = VectorizedRNG.VectorizationBase.gep(ptrx, i)
#         v = VectorizedRNG.SIMDPirates.vload(W, ptrxᵢ)
#         VectorizedRNG.SIMDPirates.vstore!(ptrxᵢ, normalcdf(v))
#         i += _W
#     end
#     if i < N
#         ptrxᵢ = VectorizedRNG.VectorizationBase.gep(ptrx, i)
#         mask = VectorizedRNG.VectorizationBase.mask(T, N & (_W - 1))
#         v = VectorizedRNG.SIMDPirates.vload(W, ptrxᵢ, mask)
#         VectorizedRNG.SIMDPirates.vstore!(ptrxᵢ, normalcdf(v), mask)
#     end
#     x
# end
function normalcdf!(x::AbstractVector{T}) where {T}
    @. x = T(0.5) * ( one(T) + erf( x * INVSQRT2 ) )
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
    Aqua.test_all(VectorizedRNG)#, ambiguities = VERSION < v"1.6-DEV")

    @test isempty(detect_unbound_args(VectorizedRNG))
    @testset "Small Crush" begin 
        # Write your own tests here.
        rngunif64 = RNGTest.wrap(local_rng(), Float64);
        res = RNGTest.smallcrushJulia(rngunif64)
        mi, ma = smallcrushextrema(res)
        @show mi, ma
        @test mi > α
        @test ma < 1 - α

        rngunif32 = RNGTest.wrap(local_rng(), Float32);
        res = RNGTest.smallcrushJulia(rngunif32)
        mi, ma = smallcrushextrema(res)
        @show mi, ma
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
        @show mi, ma
        @test mi > α
        @test ma < 1 - α

        A = zeros(13, 29);
        randn!(local_rng(), A);
        @test iszero(sum(iszero, A))
        # TODO: Support this again
        # rngnorm = RNGTest.wrap(RandNormal01(local_rng()), Float32);
        # res = RNGTest.smallcrushJulia(rngnorm)
        # mi, ma = smallcrushextrema(res)
        # @show mi, ma
        # @test mi > α
        # @test ma < 1 - α

    end
    @testset "Discontiguous in place" begin
        x = zeros(5, 117); xv = view(x, 5, :)
        rand!(local_rng(), xv)
        @test !any(iszero, xv)
        @test all(iszero, view(x, 1:4, :))
    end
    @testset "Scaled sampling" begin
        A = Matrix{Float64}(undef, 89, 100);
        randn!(local_rng(), A, VectorizedRNG.StaticInt(0), 5, 100)
        s,l = extrema(A)
        @test s < -100, l > 100
        randn!(local_rng(), A, VectorizedRNG.StaticInt(0), 100, 10)
        @test sum(A) / length(A) > 90
        randn!(local_rng(), A, VectorizedRNG.StaticInt(1), 100, 10)
        @test sum(A) / length(A) > 190
    end
end


