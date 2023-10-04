module VectorizedRNGStaticArraysExt

using VectorizedRNG: samplevector!, random_uniform, random_normal, AbstractVRNG, random_unsigned
if isdefined(Base, :get_extension)
  using StaticArraysCore
else
  using ..StaticArraysCore
end
using VectorizationBase: StaticInt
import Random

function Random.rand!(
  rng::AbstractVRNG,
  x::StaticArraysCore.MArray{<:Tuple,T},
  α::Number = StaticInt{0}(),
  β = StaticInt{0}(),
  γ = StaticInt{1}()
) where {T<:Union{Float32,Float64}}
  GC.@preserve x begin
    samplevector!(random_uniform, rng, x, α, β, γ, identity)
  end
  return x
end

function Random.randn!(
  rng::AbstractVRNG,
  x::StaticArraysCore.MArray{<:Tuple,T},
  α::Number = StaticInt{0}(),
  β = StaticInt{0}(),
  γ = StaticInt{1}()
) where {T<:Union{Float32,Float64}}
  GC.@preserve x begin
    samplevector!(random_normal, rng, x, α, β, γ, identity)
  end
  return x
end

function Random.rand!(
  rng::AbstractVRNG,
  x::StaticArraysCore.MArray{<:Tuple,UInt64}
)
  samplevector!(
    random_unsigned,
    rng,
    x,
    StaticInt{0}(),
    StaticInt{0}(),
    StaticInt{1}(),
    identity
  )
end


end
