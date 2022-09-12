
MatchingUInt(::Type{_Vec{W,Float64}}) where {W} = _Vec{W,UInt64}
MatchingUInt(::Type{Tuple{VecUnroll{N,W,Float64,Vec{W,Float64}}}}) where {N,W} = VecUnroll{N,W,UInt64,Vec{W,UInt6}}

@generated MatchingUInt(::Type{Vec{W,Float32}}) where {W} = Vec{W >>> 1,UInt64}
# @generated MatchingUInt(::Type{Tuple{Vec{W,Float32},Vararg{Vec{W,Float32},N}}}) where {N,W} = Tuple{Vec{W>>>1,UInt64},Vararg{Vec{W>>>1,UInt64},N}}
@generated MatchingUInt(::Type{VecUnroll{N,W,Float32,Vec{W,Float32}}}) where {N,W} = VecUnroll{N,W>>>1,UInt64,Vec{W>>>1,UInt64}}

# @generated MatchingFloat32(::Type{Vec{W,UInt64}}) where {W} = Vec{W<<1,Float32}
# @generated MatchingUInt32(::Type{Vec{W,UInt64}}) where {W} = Vec{W<<1,UInt32}


@inline function Random.rand(rng::AbstractVRNG{P}, ::Type{Vec{W,UInt64}}) where {W,P}
    state = getstate(rng, Val{1}(), StaticInt{W}())
    state, out = nextstate(state, Val{1}())
    storestate!(rng, state)
    data(out)[1]
end
@inline function Random.rand(rng::AbstractVRNG{P}, ::Type{VecUnroll{N,W,UInt64,Vec{W,UInt64}}}) where {W,P,N}
    state = getstate(rng, Val{P}(), StaticInt{W}())
    state, out = nextstate(state, Val{N}())
    storestate!(rng, state)
    out
end

@inline random_uniform(u::AbstractSIMD{W,UInt64}, ::Type{T}) where {W,T} = floatbitmask(u, T) - oneopenconst(T)
# @inline random_uniform(u::Vec{W,UInt64}, ::Type{Float32}) where {W} = floatbitmask(u, Float32) - oneopenconst(Float32)
# @inline random_uniform(u::Vec{W,UInt64}, ::Type{Float32}) where {W} = vsub(floatbitmask(vreinterpret(Vec{W+W,UInt32}, u), Float32), oneopenconst(Float32))
# @generated function random_uniform(u::VecUnroll{N,W,UInt64}, ::Type{T}) where {N,W,T}
#     Expr(
#         :block,
#         Expr(:meta,:inline),
#         Expr(:tuple, [Expr(:call, :random_uniform, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,Symbol(@__FILE__)), Expr(:ref, :u, n)), T) for n ∈ 1:N+1]...)
#     )
# end
@inline function Random.rand(rng::AbstractVRNG, ::Type{VecUnroll{N,W,T,Vec{W,T}}}) where {N,W,T<:Union{Float32,Float64}}
    random_uniform(rand(rng, MatchingUInt(VecUnroll{N,W,T,Vec{W,T}})), T)
end
"""
Samples uniformly from (0.0,1.0)
"""
@inline function Random.rand(rng::AbstractVRNG, ::Type{Vec{W,Float64}}) where {W}
    u = rand(rng, Vec{W,UInt64})
    random_uniform(u, Float64)
end
@generated function Random.rand(rng::AbstractVRNG, ::Type{Vec{W,T}}) where {W,T}
    L = (W * sizeof(T)) >> 3
    @assert L << 3 == W * sizeof(T)
    quote
        $(Expr(:meta,:inline))
        u = rand(rng, Vec{$L,UInt64})
        random_uniform(u, $T)
    end
end
"""
if l < u,
Samples uniformly from [l,u)
else,
Samples uniformly from (u,l]

That is, the "l" side of the interval is closed, and the "u" side is open.
"""
@inline function Random.rand(rng::AbstractVRNG, ::Type{V}, l::T, u::T) where {W,T<:Union{Float32,Float64},V <: AbstractSIMD{W,T}}
    s = u - l
    b = l - s
    u = rand(rng, MatchingUInt(Vec{W,T}))
    vfmadd(s, floatbitmask(u, T), b)
end
@inline oneopenconst(::Type{Float64}) = 0.9999999999999999
@inline oneopenconst(::Type{Float32}) = 0.99999994f0

# @generated function Random.rand(rng::AbstractVRNG, ::Type{NTuple{N,Vec{W,T}}}, l::T, u::T) where {N,W,T<:Union{Float32,Float64}}
#     quote
#         $(Expr(:meta,:inline))
#         s = u - l
#         b = l - s
#         u = rand(rng, MatchingUInt(NTuple{N,Vec{W,T}}))
#         $(Expr(:tuple, [:(vfmadd(s, floatbitmask(@inbounds(u[$n]), T), b)) for n ∈ 1:N]...))        
#     end
# end


@inline function randnormal(u1::AbstractSIMD{W,UInt64}, u2::AbstractSIMD{W,UInt64}, ::Type{T}) where {W,T<:Union{Float32,Float64}}
  s, c = randsincos(u1, T)
  r = sqrt(nlog01(u2,T))
  s * r, c * r
end
@inline function randnormal(u1::AbstractSIMD{1,UInt64}, u2::AbstractSIMD{1,UInt64}, ::Type{Float64})
  s, c = randsincos(u1(1), Float64)
  r = sqrt(nlog01(u2(1), Float64))
  Vec{1,Float64}((Core.VecElement(s * r),)), Vec{1,Float64}((Core.VecElement(c * r),))
end
@inline function randnormal(u1::UInt64, u2::UInt64, ::Type{Float64})
  s, c = randsincos(u1, Float64)
  r = sqrt(nlog01(u2, Float64))
  s*r, c*r
end

@generated function random_normal(vu::VecUnroll{Nm1,W,UInt64,Vec{W,UInt64}}, ::Type{T}) where {Nm1,W,T}
  # @assert isodd(Nm1)
  N = Nm1 + 1
  q = Expr(:block, Expr(:meta, :inline), :(u = data(vu)))
  ib = Expr(:block)
  n = 0
  if n < Nm1
    u1t = Expr(:tuple); u2t = Expr(:tuple)
    while n < Nm1
      push!(u1t.args, Expr(:ref, :u, n+1))
      push!(u2t.args, Expr(:ref, :u, n+2))
      # push!(ib.args, Expr(:(=), Expr(:tuple, Symbol(:n_,n), Symbol(:n_,n+1)), Expr(:call, :randnormal, Expr(:ref, :u, n+1), Expr(:ref, :u, n+2), T)))
      n += 2
    end
    push!(ib.args, :((sr,cr) = randnormal(VecUnroll($u1t), VecUnroll($u2t), $T)))
    push!(ib.args, :(srd = data(sr))); push!(ib.args, :(crd = data(cr)))
  end
  nout = Expr(:tuple)
  for n ∈ 1:N>>1
    push!(nout.args, Expr(:ref, :srd, n))
    push!(nout.args, Expr(:ref, :crd, n))
  end
  if n < N # then there is odd remainder
    # we split the vector in two, gen randnormal, and then recombine.
    Wl = (W << 3) ÷ sizeof(T) 
    Wh = Wl >>> 1
    t1 = Expr(:tuple); t2 = Expr(:tuple); t3 = Expr(:tuple);
    append!(t1.args, 0:Wh-1); append!(t2.args, Wh:Wl-1); append!(t3.args, 0:Wl-1)
    lm = Expr(:call, Expr(:curly, :Val, t1))
    um = Expr(:call, Expr(:curly, :Val, t2))
    cm = Expr(:call, Expr(:curly, :Val, t3))
    remq = quote
      ulast = u[$N]
      (sₗ, cᵤ) = randnormal(shufflevector(ulast, $lm), shufflevector(ulast, $um), $T)
    end
    push!(ib.args, remq)
    push!(nout.args, :(shufflevector(sₗ, cᵤ, $cm)))
  end
  push!(ib.args, :(nout = $nout))
  push!(q.args, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), ib))
  # push!(q.args, Expr(:tuple, [Symbol(:n_,n) for n ∈ 0:N-1]...))
  push!(q.args, :(VecUnroll(nout)))
  q
end

@inline Random.randn(rng::AbstractVRNG, ::Type{VecUnroll{N,W,T}}) where {N,W,T} = randn(rng, VecUnroll{N,W,T,Vec{W,T}})
@inline function Random.randn(rng::AbstractVRNG, ::Type{VecUnroll{N,W,T,Vec{W,T}}}) where {N,W,T}
    u = rand(rng, MatchingUInt(NTuple{N,Vec{W,T}}))
    random_normal(u, T)
end

@inline function random_uniform(state::AbstractState, ::Val{N}, ::Type{T}) where {N,T}
    state, u = nextstate(state, Val{N}())
    state, random_uniform(u, T)
end
@inline function random_normal(state::AbstractState, ::Val{N}, ::Type{T}) where {N,T}
    state, u = nextstate(state, Val{N}())
    state, random_normal(u, T)
end
@inline scalar_less(n::MM, i) = data(n) < i

@inline zero_pointer(A::AbstractArray) = zero_offsets(stridedpointer(A));
@inline zero_pointer(x) = x
@inline _vload(ptr::VectorizationBase.AbstractStridedPointer, args::Vararg{Any,K}) where {K} = vload(ptr, args...)
@inline _vload(x::Number, args::Vararg{Any,K}) where {K} = x

function random_sample_u2!(f::F, rng::AbstractVRNG{P}, x::AbstractArray{T}, α, β, γ) where {F,P,T}
    state = getstate(rng, Val{2}(), pick_vector_width(UInt64))
    GC.@preserve x begin
        ptrx = zero_pointer(x); ptrβ = zero_pointer(β); ptrγ = zero_pointer(γ);
        W = (pick_vector_width(T) * pick_vector_width(UInt64)) ÷ pick_vector_width(Float64); W2 = W+W
        N = length(x)
        n = MM(W, 0)
        while scalar_less(n, vadd(N, 1 - 2W))
            state, zvu2 = f(state, Val{2}(), T)
            z₁, z₂ = data(zvu2)
            x₁ = vload(ptrx, (n,)); β₁ = _vload(ptrβ, (n,)); γ₁ = _vload(ptrγ, (n,));
            x₂ = vload(ptrx, (vadd(W, n),)); β₂ = _vload(ptrβ, (vadd(W, n),)); γ₂ = _vload(ptrγ, (vadd(W, n),));
            vstore!(ptrx, α * x₁ + z₁ * γ₁ + β₁, (n,));
            vstore!(ptrx, α * x₂ + z₁ * γ₂ + β₂, (vadd(W, n),));
            n = vadd(W2, n)
        end
        m = VectorizationBase.mask(W, N)
        if scalar_less(n, vsub(N, W))
            state, zvu2 = f(state, Val{2}(), T)
            z₁, z₂ = data(zvu2)
            x₁ = vload(ptrx, (n,)); β₁ = _vload(ptrβ, (n,)); γ₁ = _vload(ptrγ, (n,));
            x₂ = vload(ptrx, (vadd(W, n),), m); β₂ = _vload(ptrβ, (vadd(W, n),), m); γ₂ = _vload(ptrγ, (vadd(W, n),), m);
            vstore!(ptrx, α * x₁ + z₁ * γ₁ + β₁, (n,));
            vstore!(ptrx, α * x₂ + z₂ * γ₂ + β₂, (vadd(W, n),), m);
        elseif scalar_less(n, N)
            state, zvu1 = f(state, Val{1}(), T)
            (z₁,) = data(zvu1)
            x₁ = vload(ptrx, (n,), m); β₁ = _vload(ptrβ, (n,), m); γ₁ = _vload(ptrγ, (n,), m);
            vstore!(ptrx, α * x₁ + z₁ * γ₁ + β₁, (n,), m);
        end
        storestate!(rng, state)
    end # GC preserve
    x
end
function random_sample_u2!(f::F, rng::AbstractVRNG{P}, x::AbstractArray{T}, ::StaticInt{0}, β, γ) where {F,P,T}
    state = getstate(rng, Val{2}(), pick_vector_width(UInt64))
    GC.@preserve x begin
        ptrx = zero_pointer(x); ptrβ = zero_pointer(β); ptrγ = zero_pointer(γ);
        W = (pick_vector_width(T) * pick_vector_width(UInt64)) ÷ pick_vector_width(Float64); W2 = W+W
        N = length(x)
        n = MM(W, 0)
        while scalar_less(n, vadd(N, 1 - 2W))
            state, zvu2 = f(state, Val{2}(), T)
            (z₁,z₂) = data(zvu2)
            β₁ = _vload(ptrβ, (n,)); γ₁ = _vload(ptrγ, (n,));
            β₂ = _vload(ptrβ, (vadd(W, n),)); γ₂ = _vload(ptrγ, (vadd(W, n),));
            vstore!(ptrx, z₁ * γ₁ + β₁, (n,));
            vstore!(ptrx, z₂ * γ₂ + β₂, (vadd(W, n),));
            n = vadd(W2, n)
        end
        m = VectorizationBase.mask(W, N)
        if scalar_less(n, vsub(N, W))
            state, zvu2 = f(state, Val{2}(), T)
            (z₁,z₂) = data(zvu2)
            β₁ = _vload(ptrβ, (n,)); γ₁ = _vload(ptrγ, (n,));
            β₂ = _vload(ptrβ, (vadd(W, n),), m); γ₂ = _vload(ptrγ, (vadd(W, n),), m);
            vstore!(ptrx, z₁ * γ₁ + β₁, (n,));
            vstore!(ptrx, z₂ * γ₂ + β₂, (vadd(W, n),), m);
        elseif scalar_less(n, N)
            state, zvu1 = f(state, Val{1}(), T)
            (z₁,) = data(zvu1)
            β₁ = _vload(ptrβ, (n,), m); γ₁ = _vload(ptrγ, (n,), m);
            vstore!(ptrx, z₁ * γ₁ + β₁, (n,), m);
        end
        storestate!(rng, state)
    end # GC preserve
    # @show state getstate(rng, Val{2}(), Val{W64}())
    # @assert state === getstate(rng, Val{2}(), Val{W64}())
    x
end

function Random.rand!(
    rng::AbstractVRNG, x::AbstractArray{T}, α::Number = StaticInt{0}(), β = StaticInt{0}(), γ = StaticInt{1}()
) where {T <: Union{Float32,Float64}}
  random_sample_u2!(random_uniform, rng, x, α, β, γ)
end
function Random.randn!(
    rng::AbstractVRNG, x::AbstractArray{T}, α::Number = StaticInt{0}(), β = StaticInt{0}(), γ = StaticInt{1}()
) where {T<:Union{Float32,Float64}}
  random_sample_u2!(random_normal, rng, x, α, β, γ)
end
@inline function random_unsigned(state::AbstractState, ::Val{N}, ::Type{T}) where {N,T}
  nextstate(state, Val{N}())
end
function Random.rand!(rng::AbstractVRNG, x::AbstractArray{UInt64})
  random_sample_u2!(random_unsigned, rng, x, StaticInt{0}(), StaticInt{0}(), StaticInt{1}())
end

using StaticArraysCore, StrideArraysCore
function Random.rand!(rng::AbstractVRNG, x::StaticArraysCore.MArray{<:Tuple,T}) where {T<:Union{Float32,Float64}}
  GC.@preserve x begin
    random_sample_u2!(random_uniform, rng, PtrArray(x), α, β, γ)
  end
  return x
end
function Random.rand!(rng::AbstractVRNG, x::SA) where {S<:Tuple,T<:Union{Float32,Float64},SA<:StaticArraysCore.StaticArray{S,T}}
  a = MArray{S,UInt64}(undef)
  GC.@preserve a begin
    random_sample_u2!(random_uniform, rng, PtrArray(a), α, β, γ)
  end
  x .= a
end
function Random.randn!(rng::AbstractVRNG, x::StaticArraysCore.MArray{<:Tuple,T}) where {T<:Union{Float32,Float64}}
  GC.@preserve x begin
    random_sample_u2!(random_normal, rng, PtrArray(x), α, β, γ)
  end
  return x
end
function Random.randn!(rng::AbstractVRNG, x::SA) where {S<:Tuple,T<:Union{Float32,Float64},SA<:StaticArraysCore.StaticArray{S,T}}
  a = MArray{S,UInt64}(undef)
  GC.@preserve a begin
    random_sample_u2!(random_normal, rng, PtrArray(a), α, β, γ)
  end
  x .= a
end

function Random.rand!(rng::AbstractVRNG, x::StaticArraysCore.MArray{<:Tuple,UInt64})
  random_sample_u2!(random_unsigned, rng, x, StaticInt{0}(), StaticInt{0}(), StaticInt{1}())
end
function Random.rand!(rng::AbstractVRNG, x::SA) where {S<:Tuple,SA<:StaticArraysCore.StaticArray{S,UInt64}}
  a = MArray{S,UInt64}(undef)
  GC.@preserve a begin
    random_sample_u2!(random_unsigned, rng, PtrArray(a), StaticInt{0}(), StaticInt{0}(), StaticInt{1}())
  end
  x .= a
end



Random.rand(rng::AbstractVRNG, d1::Integer, dims::Vararg{Integer,N}) where {N} = rand!(rng, Array{Float64}(undef, d1, dims...))
Random.randn(rng::AbstractVRNG, d1::Integer, dims::Vararg{Integer,N}) where {N} = randn!(rng, Array{Float64}(undef, d1, dims...))
# Random.randexp(rng::AbstractVRNG, d1::Integer, dims::Vararg{Integer,N}) where {N} = randexp!(rng, Array{Float64}(undef, d1, dims...))


struct Buffer256{T} <: DenseVector{T}
    ptr::Ptr{T}
end
Base.pointer(b::Buffer256) = b.ptr
Base.length(::Buffer256) = 256
Base.size(::Buffer256) = (256,)
Base.getindex(b::Buffer256, i::Int) = vload(stridedpointer(b), (i,))
Base.strides(::Buffer256) = (1,)
VectorizationBase.ArrayInterface.contiguous_axis(::Type{<:Buffer256}) = VectorizationBase.One()
VectorizationBase.ArrayInterface.contiguous_batch_size(::Type{<:Buffer256}) = VectorizationBase.Zero()
VectorizationBase.ArrayInterface.stride_rank(::Type{<:Buffer256}) = (VectorizationBase.One(),)


@inline function Random.rand(rng::AbstractVRNG)
    i = getrand64counter(rng)
    b = randbuffer64(rng)
    setrand64counter!(rng, i + 0x01)
    iszero(i) && rand!(rng, b)
    vloadu(pointer(b), VectorizationBase.LazyMulAdd{8,0}(i % UInt32))
end

# Box-Muller for scalars is probably only faster with AVX512, so we use Ziggurat if we don't have it.
@inline function randn_scalar(rng::AbstractVRNG, ::VectorizationBase.True)
    i = getrandn64counter(rng)
    b = randnbuffer64(rng)
    setrandn64counter!(rng, i + 0x01)
    iszero(i) && randn!(rng, b)
    vloadu(pointer(b), VectorizationBase.LazyMulAdd{8,0}(i % UInt32))
end
@inline randn_scalar(rng::AbstractVRNG, ::VectorizationBase.False) = Random._randn(rng, rand(rng, Random.UInt52Raw{UInt64}()))
@inline Random.randn(rng::AbstractVRNG) = randn_scalar(rng, VectorizationBase.has_feature(Val(:x86_64_avx512f)))

@inline function Random.rand(rng::AbstractVRNG, ::Random.SamplerType{UInt64})
  state, u = nextstate(getstate(rng))
  storestate!(rng, state)
  return u
end
# @inline function Random.rand(rng::AbstractVRNG, ::Random.SamplerType{UInt64})
#     i = getrandu64counter(rng)
#     b = randubuffer64(rng)
#     setrandu64counter!(rng, i + 0x01)
#     iszero(i) && rand!(rng, b)
#     vloadu(pointer(b), VectorizationBase.LazyMulAdd{8,0}(i % UInt32))
# end
@inline Random.rand(rng::AbstractVRNG, ::Random.UInt52Raw{UInt64}) = Random.rand(rng, Random.SamplerType{UInt64}())
for T ∈ [:Int8,:UInt8,:Int16,:UInt16,:Int32,:UInt32,:Int64]
  @eval @inline Random.rand(rng::AbstractVRNG, ::Random.SamplerType{$T}) = Random.rand(rng, Random.SamplerType{UInt64}()) % $T
end
@inline function Random.rand(rng::AbstractVRNG, ::Random.SamplerType{T}) where {T<:Union{UInt128,Int128}}
  i = getrandu64counter(rng)
  b = randubuffer64(rng)
  if i == 0xff
    rand!(rng, b)
    setrandu64counter!(rng, 0x02)
  else
    setrandu64counter!(rng, 0x02 + i)
    iszero(i) && rand!(rng, b)
  end
    vloadu(Base.unsafe_convert(Ptr{T}, pointer(b)), VectorizationBase.LazyMulAdd{8,0}(i % UInt32))
end
Random.rng_native_52(::AbstractVRNG) = UInt64

