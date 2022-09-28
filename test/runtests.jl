using VectorizedRNG
using Test

using InteractiveUtils: versioninfo
versioninfo(verbose=true)

using RNGTest, Random, SpecialFunctions, Aqua, Distributions

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
#     W = VectorizedRNG.VectorizationBase.pick_vector_width(T)
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
normalcdf(x::T) where {T} = T(0.5) * ( one(T) + erf( x * INVSQRT2 ) )
function normalcdf!(x::AbstractVector{T}) where {T}
  @. x = normalcdf(x)
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

function test_serial_rng(f)
  res = RNGTest.smallcrushJulia(f)
  mi, ma = smallcrushextrema(res)
  @show mi, ma
  @test mi > α
  @test ma < 1 - α
end

@testset "VectorizedRNG.jl" begin
  Aqua.test_all(VectorizedRNG)#, ambiguities = VERSION < v"1.6-DEV")
  VectorizedRNG.seed!(33)
  @test isempty(detect_unbound_args(VectorizedRNG))
  @testset "Small Crush" begin 
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

    rngnorm = RNGTest.wrap(RandNormal01(local_rng()), Float64);
    res = RNGTest.smallcrushJulia(rngnorm)
    mi, ma = smallcrushextrema(res)
    @show mi, ma
    @test mi > α
    @test ma < 1 - α

    mxoshift = VectorizedRNG.MutableXoshift(3)
    rngunif64 = RNGTest.wrap(mxoshift, Float64);
    res = RNGTest.smallcrushJulia(rngunif64)
    mi, ma = smallcrushextrema(res)
    @show mi, ma
    @test mi > α
    @test ma < 1 - α


    rngunif32 = RNGTest.wrap(mxoshift, Float32);
    res = RNGTest.smallcrushJulia(rngunif32)
    mi, ma = smallcrushextrema(res)
    @show mi, ma
    @test mi > α
    @test ma < 1 - α

    rngnorm = RNGTest.wrap(RandNormal01(mxoshift), Float64);
    res = RNGTest.smallcrushJulia(rngnorm)
    mi, ma = smallcrushextrema(res)
    @show mi, ma
    @test mi > α
    @test ma < 1 - α

    A = zeros(13, 29);
    randn!(local_rng(), A);
    @test iszero(sum(iszero, A))
    A .= 0;
    randn!(mxoshift, A);
    @test iszero(sum(iszero, A))
    A32 = zeros(Float32, 13, 29);
    randn!(local_rng(), A32);
    @test iszero(sum(iszero, A32))
    A32 .= 0;
    randn!(mxoshift, A32);
    @test iszero(sum(iszero, A32))

    # TODO: Support this again
    # rngnorm = RNGTest.wrap(RandNormal01(local_rng()), Float32);
    # res = RNGTest.smallcrushJulia(rngnorm)
    # mi, ma = smallcrushextrema(res)
    # @show mi, ma
    # @test mi > α
    # @test ma < 1 - α

    # scalar mode tests
    test_serial_rng(() -> rand(local_rng()))
    test_serial_rng(() -> normalcdf(randn(local_rng())))
    test_serial_rng(() -> cdf(Gamma(), rand(local_rng(), Gamma())))
    test_serial_rng(() -> VectorizedRNG.floatbitmask((rand(local_rng(), UInt128) >> 64) % UInt64, Float64) - VectorizedRNG.oneopenconst(Float64))
    
  end
  @testset "Discontiguous in place" begin
    x = zeros(5, 117); xv = view(x, 5, :)
    rand!(local_rng(), xv)
    @test !any(iszero, xv)
    @test all(iszero, view(x, 1:4, :))
  end
  @testset "Scaled sampling" begin
    A = Matrix{Float64}(undef, 89, 100);
    randn!(local_rng(), A, VectorizedRNG.StaticInt(0), 5, 100);
    s,l = extrema(A)
    @test s < -100
    @test l > 100
    randn!(local_rng(), A, VectorizedRNG.StaticInt(0), 100, 10);
    @test 90 < sum(A) / length(A) < 110
    randn!(local_rng(), A, VectorizedRNG.StaticInt(1), 100, 10);
    @test 190 < sum(A) / length(A) < 210
  end
  @testset "Correct Sigma" begin
    for T = (Float32,Float64)
      x = Vector{T}(undef, 15);
      s::Float64 = 0.0
      N = 10_000;
      vrng = local_rng()
      σ = 0.5
      for i = 1:N
        randn!(vrng, x, static(0), static(0), σ)
        s += std(x)
      end
      s /= N
      @test s ≈ σ rtol = 1e-1
    end
  end
end


