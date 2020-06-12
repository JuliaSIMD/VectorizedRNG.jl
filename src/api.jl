MatchingUInt(::Type{Vec{W,Float64}}) where {W} = Vec{W,UInt64}
MatchingUInt(::Type{NTuple{N,Vec{W,Float64}}}) where {N,W} = NTuple{N,Vec{W,UInt64}}

@generated MatchingUInt(::Type{Vec{W,Float32}}) where {W} = Vec{cld(W,2),UInt64}
@generated MatchingUInt(::Type{NTuple{N,Vec{W,Float32}}}) where {N,W} = NTuple{N,Vec{cld(W,2),UInt64}}

@generated MatchingFloat32(::Type{Vec{W,UInt64}}) where {W} = Vec{W<<1,Float32}


@inline function Random.rand(rng::AbstractVRNG{P}, ::Type{Vec{W,UInt64}}) where {W,P}
    state = getstate(rng, Val{1}(), Val{W}())
    state, (out,) = nextstate(state, Val{1}())
    storestate!(rng, state)
    out
end
@inline function Random.rand(rng::AbstractVRNG{P}, ::Type{NTuple{N,Vec{W,UInt64}}}) where {W,P,N}
    state = getstate(rng, Val{P}(), Val{W}())
    state, out = nextstate(state, Val{N}())
    storestate!(rng, state)
    out
end

@inline random_uniform(u::Vec{W,UInt64}, ::Type{T}) where {W,T} = vsub(mask(u, T), oneopenconst(T))
@generated function random_uniform(u::NTuple{N,Vec{W,UInt64}}, ::Type{T}) where {N,W,T}
    Expr(
        :block,
        Expr(:meta,:inline),
        Expr(:tuple, [Expr(:call, :random_uniform, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,Symbol(@__FILE__)), Expr(:ref, :u, n)), T) for n ∈ 1:N]...)
    )
end
@inline function Random.rand(rng::AbstractVRNG, ::Type{NTuple{N,Vec{W,T}}}) where {N,W,T<:Union{Float32,Float64}}
    random_uniform(rand(rng, MatchingUInt(NTuple{N,Vec{W,T}})), T)
end
"""
Samples uniformly from (0.0,1.0)
"""
@inline function Random.rand(rng::AbstractVRNG, ::Type{Vec{W,T}}) where {W,T}
    u = rand(rng, Vec{W,UInt64})
    random_uniform(u, T)
end
"""
if l < u,
Samples uniformly from [l,u)
else,
Samples uniformly from (u,l]

That is, the "l" side of the interval is closed, and the "u" side is open.
"""
@inline function Random.rand(rng::AbstractVRNG, ::Type{Vec{W,T}}, l::T, u::T) where {W,T<:Union{Float32,Float64}}
    s = u - l
    b = l - s
    u = rand(rng, MatchingUInt(Vec{W,T}))
    vfmadd(s, mask(u, T), b)
end
@inline oneopenconst(::Type{Float64}) = 0.9999999999999999
@inline oneopenconst(::Type{Float32}) = 0.99999994f0

@generated function Random.rand(rng::AbstractVRNG, ::Type{NTuple{N,Vec{W,T}}}, l::T, u::T) where {N,W,T<:Union{Float32,Float64}}
    quote
        $(Expr(:meta,:inline))
        s = u - l
        b = l - s
        u = rand(rng, MatchingUInt(NTuple{N,Vec{W,T}}))
        $(Expr(:tuple, [:(vfmadd(s, mask(@inbounds(u[$n]), T), b)) for n ∈ 1:N]...))        
    end
end


@inline function randnormal(u1::Vec{W,UInt64}, u2::Vec{W,UInt64}, ::Type{T}) where {W,T<:Union{Float32,Float64}}
    s, c = randsincos(u1, T)
    r = vsqrt(nlog01(u2,T))
    vmul(s,r), vmul(c,r)
end

@generated function random_normal(u::NTuple{N,Vec{W,UInt64}}, ::Type{T}) where {N,W,T}
    q = Expr(:block, Expr(:meta, :inline))
    ib = Expr(:block)
    n = 0
    while n < N - 1
        push!(ib.args, Expr(:(=), Expr(:tuple, Symbol(:n_,n), Symbol(:n_,n+1)), Expr(:call, :randnormal, Expr(:ref, :u, n+1), Expr(:ref, :u, n+2), T)))
        n += 2
    end
    if n < N # then there is odd remainder
        # we split the vector in two, gen randnormal, and then recombine.
        Wh = W >>> 1
        lm = Expr(:call, Expr(:curly, :Val, Expr(:tuple, collect(0:Wh-1)...)))
        um = Expr(:call, Expr(:curly, :Val, Expr(:tuple, collect(Wh:W-1)...)))
        cm = Expr(:call, Expr(:curly, :Val, Expr(:tuple, collect(0:W-1)...)))
        remq = quote
            ulast = u[$N]
            (sₗ, cᵤ) = randnormal(shufflevector(ulast, $lm), shufflevector(ulast, $um), $T)
            $(Symbol(:n_,N-1)) = shufflevector(sₗ, cᵤ, $cm)
        end
        push!(ib.args, remq)
    end
    push!(q.args, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), ib))
    push!(q.args, Expr(:tuple, [Symbol(:n_,n) for n ∈ 0:N-1]...))
    q
end

@inline function Random.randn(rng::AbstractVRNG, ::Type{NTuple{N,Vec{W,T}}}) where {N,W,T}
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

function random_sample!(f::typeof(random_uniform), rng::AbstractVRNG{P}, x::AbstractArray{Float64}) where {P}
    state = getstate(rng, Val{2}(), Val{W64}())
    GC.@preserve x begin
        ptrx = stridedpointer(x)
        W = VectorizationBase.pick_vector_width(Float64)
        N = length(x)
        n = _MM(VectorizationBase.pick_vector_width_val(Float64), 0)
        while VectorizationBase.scalar_less(n, vadd(N, 1 - 2W))
            state, (z₁,z₂) = f(state, Val{2}(), Float64)
            vstore!(ptrx, z₁, (n,)); n = vadd(W, n)
            vstore!(ptrx, z₂, (n,)); n = vadd(W, n)
        end
        mask = VectorizationBase.mask(Val{W64}(), N)
        if VectorizationBase.scalar_less(n, vsub(N, W))
            state, (z₁,z₂) = f(state, Val{2}(), Float64)
            vstore!(ptrx, z₁, (n,)); n = vadd(W, n)
            vstore!(ptrx, z₂, (n,), mask);
        elseif VectorizationBase.scalar_less(n, N)
            vstate, (z₁,) = f(state, Val{1}(), Float64)
            vstore!(ptrx, z₁, (n,), mask);
        end
        storestate!(rng, state)
    end # GC preserve
    x
end
function random_sample!(f::F, rng::AbstractVRNG{P}, x::AbstractArray{Float64}) where {F,P}
    state = getstate(rng, Val{P}(), Val{W64}())
    GC.@preserve x begin
        ptrx = stridedpointer(x)
        W = VectorizationBase.pick_vector_width(Float64)
        N = length(x)
        n = _MM(VectorizationBase.pick_vector_width_val(Float64), 0)
        while VectorizationBase.scalar_less(n, vadd(N, 1 - 4W))
            state, (z₁,z₂,z₃,z₄) = f(state, Val{4}(), Float64)
            vstore!(ptrx, z₁, (n,)); n = vadd(W, n)
            vstore!(ptrx, z₂, (n,)); n = vadd(W, n)
            vstore!(ptrx, z₃, (n,)); n = vadd(W, n)
            vstore!(ptrx, z₄, (n,)); n = vadd(W, n) 
        end
        mask = VectorizationBase.mask(Val{W64}(), N)
        if VectorizationBase.scalar_less(n, vsub(N, 3W))
            state, (z₁,z₂,z₃,z₄) = f(state, Val{4}(), Float64)
            vstore!(ptrx, z₁, (n,)); n = vadd(W, n)
            vstore!(ptrx, z₂, (n,)); n = vadd(W, n)
            vstore!(ptrx, z₃, (n,)); n = vadd(W, n)
            vstore!(ptrx, z₄, (n,), mask);
        elseif VectorizationBase.scalar_less(n, vsub(N, 2W))
            state, (z₁,z₂,z₃) = f(state, Val{3}(), Float64)
            vstore!(ptrx, z₁, (n,)); n = vadd(W, n)
            vstore!(ptrx, z₂, (n,)); n = vadd(W, n)
            vstore!(ptrx, z₃, (n,), mask);
        elseif VectorizationBase.scalar_less(n, vsub(N, W))
            state, (z₁,z₂) = f(state, Val{2}(), Float64)
            vstore!(ptrx, z₁, (n,)); n = vadd(W, n)
            vstore!(ptrx, z₂, (n,), mask);
        elseif VectorizationBase.scalar_less(n, N)
            vstate, (z₁,) = f(state, Val{1}(), Float64)
            vstore!(ptrx, z₁, (n,), mask);
        end        
        storestate!(rng, state)
    end # GC preserve
    x
end
function Random.rand!(rng::AbstractVRNG, x::AbstractArray{Float64})
    random_sample!(random_uniform, rng, x)
end
function Random.randn!(rng::AbstractVRNG, x::AbstractArray{Float64})
    random_sample!(random_normal, rng, x)
end
@inline function random_unsigned(state::AbstractState, ::Val{N}, ::Type{T}) where {N,T}
    nextstate(state, Val{N}())
end
function Random.rand!(rng::AbstractVRNG, x::AbstractArray{UInt64})
    random_sample!(random_unsigned, rng, x)
end

Random.rand(rng::AbstractVRNG, d1::Integer, dims::Vararg{Integer,N}) where {N} = rand!(rng, Array{Float64}(undef, d1, dims...))
Random.randn(rng::AbstractVRNG, d1::Integer, dims::Vararg{Integer,N}) where {N} = randn!(rng, Array{Float64}(undef, d1, dims...))
# Random.randexp(rng::AbstractVRNG, d1::Integer, dims::Vararg{Integer,N}) where {N} = randexp!(rng, Array{Float64}(undef, d1, dims...))


