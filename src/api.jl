MatchingUInt(::Type{_Vec{W,Float64}}) where {W} = _Vec{W,UInt64}
MatchingUInt(::Type{Tuple{VecUnroll{N,W,Float64,Vec{W,Float64}}}}) where {N,W} = VecUnroll{N,W,UInt64,Vec{W,UInt6}}

@generated MatchingUInt(::Type{Vec{W,Float32}}) where {W} = Vec{W >>> 1,UInt64}
# @generated MatchingUInt(::Type{Tuple{Vec{W,Float32},Vararg{Vec{W,Float32},N}}}) where {N,W} = Tuple{Vec{W>>>1,UInt64},Vararg{Vec{W>>>1,UInt64},N}}
@generated MatchingUInt(::Type{VecUnroll{N,W,Float32,Vec{W,Float32}}}) where {N,W} = VecUnroll{N,W>>>1,UInt64,Vec{W>>>1,UInt64}}

# @generated MatchingFloat32(::Type{Vec{W,UInt64}}) where {W} = Vec{W<<1,Float32}
# @generated MatchingUInt32(::Type{Vec{W,UInt64}}) where {W} = Vec{W<<1,UInt32}


@inline function Random.rand(rng::AbstractVRNG{P}, ::Type{Vec{W,UInt64}}) where {W,P}
    state = getstate(rng, Val{1}(), Val{W}())
    state, out = nextstate(state, Val{1}())
    storestate!(rng, state)
    out.data[1]
end
@inline function Random.rand(rng::AbstractVRNG{P}, ::Type{VecUnroll{N,W,UInt64,Vec{W,UInt64}}}) where {W,P,N}
    state = getstate(rng, Val{P}(), Val{W}())
    state, out = nextstate(state, Val{N}())
    storestate!(rng, state)
    out
end

@inline random_uniform(u::AbstractSIMD{W,UInt64}, ::Type{T}) where {W,T} = mask(u, T) - oneopenconst(T)
# @inline random_uniform(u::Vec{W,UInt64}, ::Type{Float32}) where {W} = mask(u, Float32) - oneopenconst(Float32)
# @inline random_uniform(u::Vec{W,UInt64}, ::Type{Float32}) where {W} = vsub(mask(vreinterpret(Vec{W+W,UInt32}, u), Float32), oneopenconst(Float32))
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
    vfmadd(s, mask(u, T), b)
end
@inline oneopenconst(::Type{Float64}) = 0.9999999999999999
@inline oneopenconst(::Type{Float32}) = 0.99999994f0

# @generated function Random.rand(rng::AbstractVRNG, ::Type{NTuple{N,Vec{W,T}}}, l::T, u::T) where {N,W,T<:Union{Float32,Float64}}
#     quote
#         $(Expr(:meta,:inline))
#         s = u - l
#         b = l - s
#         u = rand(rng, MatchingUInt(NTuple{N,Vec{W,T}}))
#         $(Expr(:tuple, [:(vfmadd(s, mask(@inbounds(u[$n]), T), b)) for n ∈ 1:N]...))        
#     end
# end


@inline function randnormal(u1::AbstractSIMD{W,UInt64}, u2::AbstractSIMD{W,UInt64}, ::Type{T}) where {W,T<:Union{Float32,Float64}}
    s, c = randsincos(u1, T)
    r = sqrt(nlog01(u2,T))
    s * r, c * r
end

@generated function random_normal(vu::VecUnroll{Nm1,W,UInt64,Vec{W,UInt64}}, ::Type{T}) where {Nm1,W,T}
    # @assert isodd(Nm1)
    N = Nm1 + 1
    q = Expr(:block, Expr(:meta, :inline), :(u = vu.data))
    ib = Expr(:block)
    n = 0
    u1t = Expr(:tuple); u2t = Expr(:tuple)
    while n < N - 1
        push!(u1t.args, Expr(:ref, :u, n+1))
        push!(u2t.args, Expr(:ref, :u, n+2))
        # push!(ib.args, Expr(:(=), Expr(:tuple, Symbol(:n_,n), Symbol(:n_,n+1)), Expr(:call, :randnormal, Expr(:ref, :u, n+1), Expr(:ref, :u, n+2), T)))
        n += 2
    end
    push!(ib.args, :((sr,cr) = randnormal(VecUnroll($u1t), VecUnroll($u2t), $T)))
    push!(ib.args, :(srd = sr.data)); push!(ib.args, :(crd = cr.data))
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
@inline scalar_less(n::MM, i) = n.i < i
function random_sample_u2!(f::F, rng::AbstractVRNG{P}, x::AbstractArray{T}, α, β, γ) where {F,P,T}
    state = getstate(rng, Val{2}(), Val{W64}())
    GC.@preserve x begin
        ptrx = zero_offsets(stridedpointer(x)); ptrβ = zero_offsets(stridedpointer(β)); ptrγ = zero_offsets(stridedpointer(γ));
        W = VectorizationBase.pick_vector_width(T); W2 = W+W
        Wval = VectorizationBase.pick_vector_width_val(T)
        N = length(x)
        n = MM(Wval, 0)
        while scalar_less(n, vadd(N, 1 - 2W))
            state, zvu2 = f(state, Val{2}(), T)
            z₁, z₂ = zvu2.data
            x₁ = vload(ptrx, (n,)); β₁ = vload(ptrβ, (n,)); γ₁ = vload(ptrγ, (n,));
            x₂ = vload(ptrx, (vadd(W, n),)); β₂ = vload(ptrβ, (vadd(W, n),)); γ₂ = vload(ptrγ, (vadd(W, n),));
            vstore!(ptrx, α * x₁ + z₁ * γ₁ + β₁, (n,));
            vstore!(ptrx, α * x₂ + z₁ * γ₂ + β₂, (vadd(W, n),));
            n = vadd(W2, n)
        end
        mask = VectorizationBase.mask(Wval, N)
        if scalar_less(n, vsub(N, W))
            state, zvu2 = f(state, Val{2}(), T)
            z₁, z₂ = zvu2.data
            x₁ = vload(ptrx, (n,)); β₁ = vload(ptrβ, (n,)); γ₁ = vload(ptrγ, (n,));
            x₂ = vload(ptrx, (vadd(W, n),), mask); β₂ = vload(ptrβ, (vadd(W, n),), mask); γ₂ = vload(ptrγ, (vadd(W, n),), mask);
            vstore!(ptrx, α * x₁ + z₁ * γ₁ + β₁, (n,));
            vstore!(ptrx, α * x₂ + z₂ * γ₂ + β₂, (vadd(W, n),), mask);
        elseif scalar_less(n, N)
            state, zvu1 = f(state, Val{1}(), T)
            (z₁,) = zvu1.data
            x₁ = vload(ptrx, (n,), mask); β₁ = vload(ptrβ, (n,), mask); γ₁ = vload(ptrγ, (n,), mask);
            vstore!(ptrx, α * x₁ + z₁ * γ₁ + β₁, (n,), mask);
        end
        storestate!(rng, state)
    end # GC preserve
    x
end
function random_sample_u4!(f::F, rng::AbstractVRNG{P}, x::AbstractArray{T}, α, β, γ) where {F,P,T}
    state = getstate(rng, Val{P}(), Val{W64}())
    GC.@preserve x begin
        ptrx = zero_offsets(stridedpointer(x)); ptrβ = zero_offsets(stridedpointer(β)); ptrγ = zero_offsets(stridedpointer(γ));
        W = VectorizationBase.pick_vector_width(T); W2 = W+W; W3 = W2+W; W4 = W2+W2;
        Wval = VectorizationBase.pick_vector_width_val(T)
        N = length(x)
        n = MM(Wval, 0)
        while scalar_less(n, vadd(N, 1 - 4W))
            state, zvu4 = f(state, Val{4}(), T)
            (z₁,z₂,z₃,z₄) = zvu4.data
            x₁ = vload(ptrx, (n,)); β₁ = vload(ptrβ, (n,)); γ₁ = vload(ptrγ, (n,));
            x₂ = vload(ptrx, (vadd(W, n),)); β₂ = vload(ptrβ, (vadd(W, n),)); γ₂ = vload(ptrγ, (vadd(W, n),));
            x₃ = vload(ptrx, (vadd(W2, n),)); β₃ = vload(ptrβ, (vadd(W2, n),)); γ₃ = vload(ptrγ, (vadd(W2, n),));
            x₄ = vload(ptrx, (vadd(W3, n),)); β₄ = vload(ptrβ, (vadd(W3, n),)); γ₄ = vload(ptrγ, (vadd(W3, n),));
            vstore!(ptrx, α * x₁ + z₁ * γ₁ + β₁, (n,));
            vstore!(ptrx, α * x₂ + z₂ * y₂ + β₂, (vadd(W,n),));
            vstore!(ptrx, α * x₃ + z₃ * γ₃ + β₃, (vadd(W2,n),));
            vstore!(ptrx, α * x₄ + z₄ * γ₄ + β₄, (vadd(W3,n),));
            n = vadd(W4, n) 
        end
        mask = VectorizationBase.mask(Wval, N)
        if scalar_less(n, vsub(N, 3W))
            state, zvu4 = f(state, Val{4}(), T)
            (z₁,z₂,z₃,z₄) = zvu4.data
            x₁ = vload(ptrx, (n,)); β₁ = vload(ptrβ, (n,)); γ₁ = vload(ptrγ, (n,));
            x₂ = vload(ptrx, (vadd(W, n),)); β₂ = vload(ptrβ, (vadd(W, n),)); γ₂ = vload(ptrγ, (vadd(W, n),));
            x₃ = vload(ptrx, (vadd(W2, n),)); β₃ = vload(ptrβ, (vadd(W2, n),)); γ₃ = vload(ptrγ, (vadd(W2, n),));
            x₄ = vload(ptrx, (vadd(W3, n),), mask); β₄ = vload(ptrβ, (vadd(W3, n),), mask); γ₄ = vload(ptrγ, (vadd(W3, n),), mask);
            vstore!(ptrx, α * x₁ + z₁ * γ₁ + β₁, (n,));
            vstore!(ptrx, α * x₂ + z₂ * y₂ + β₂, (vadd(W,n),));
            vstore!(ptrx, α * x₃ + z₃ * γ₃ + β₃, (vadd(W2,n),));
            vstore!(ptrx, α * x₄ + z₄ * γ₄ + β₄, (vadd(W3,n),), mask);
        elseif scalar_less(n, vsub(N, 2W))
            state, zvu3 = f(state, Val{3}(), T)
            (z₁,z₂,z₃) = zvu3.data
            x₁ = vload(ptrx, (n,)); β₁ = vload(ptrβ, (n,)); γ₁ = vload(ptrγ, (n,));
            x₂ = vload(ptrx, (vadd(W, n),)); β₂ = vload(ptrβ, (vadd(W, n),)); γ₂ = vload(ptrγ, (vadd(W, n),));
            x₃ = vload(ptrx, (vadd(W2, n),), mask); β₃ = vload(ptrβ, (vadd(W2, n),), mask); γ₃ = vload(ptrγ, (vadd(W2, n),), mask);
            vstore!(ptrx, α * x₁ + z₁ * γ₁ + β₁, (n,));
            vstore!(ptrx, α * x₂ + z₂ * y₂ + β₂, (vadd(W,n),));
            vstore!(ptrx, α * x₃ + z₃ * γ₃ + β₃, (vadd(W2,n),), mask);
        elseif scalar_less(n, vsub(N, W))
            state, zvu2 = f(state, Val{2}(), T)
            (z₁,z₂) = zvu2.data
            x₁ = vload(ptrx, (n,)); β₁ = vload(ptrβ, (n,)); γ₁ = vload(ptrγ, (n,));
            x₂ = vload(ptrx, (vadd(W, n),), mask); β₂ = vload(ptrβ, (vadd(W, n),), mask); γ₂ = vload(ptrγ, (vadd(W, n),), mask);
            vstore!(ptrx, α * x₁ + z₁ * γ₁ + β₁, (n,));
            vstore!(ptrx, α * x₂ + z₂ * γ₂ + β₂, (vadd(W,n),), mask);
        elseif scalar_less(n, N)
            state, zvu1 = f(state, Val{1}(), T)
            (z₁,) = zvu1.data
            x₁ = vload(ptrx, (n,), mask); β₁ = vload(ptrβ, (n,), mask); γ₁ = vload(ptrγ, (n,), mask);
            vstore!(ptrx, α * x₁ + z₁ * γ₁ + β₁, (n,), mask);
        end        
        storestate!(rng, state)
    end # GC preserve
    x
end
function random_sample_u2!(f::F, rng::AbstractVRNG{P}, x::AbstractArray{T}, ::StaticInt{0}, β, γ) where {F,P,T}
    state = getstate(rng, Val{2}(), Val{W64}())
    GC.@preserve x begin
        ptrx = zero_offsets(stridedpointer(x)); ptrβ = zero_offsets(stridedpointer(β)); ptrγ = zero_offsets(stridedpointer(γ));
        W = VectorizationBase.pick_vector_width(T); W2 = W+W
        Wval = VectorizationBase.pick_vector_width_val(T)
        N = length(x)
        n = MM(Wval, 0)
        while scalar_less(n, vadd(N, 1 - 2W))
            state, zvu2 = f(state, Val{2}(), T)
            (z₁,z₂) = zvu2.data
            β₁ = vload(ptrβ, (n,)); γ₁ = vload(ptrγ, (n,));
            β₂ = vload(ptrβ, (vadd(W, n),)); γ₂ = vload(ptrγ, (vadd(W, n),));
            vstore!(ptrx, z₁ * γ₁ + β₁, (n,));
            vstore!(ptrx, z₂ * γ₂ + β₂, (vadd(W, n),));
            n = vadd(W2, n)
        end
        mask = VectorizationBase.mask(Wval, N)
        if scalar_less(n, vsub(N, W))
            state, zvu2 = f(state, Val{2}(), T)
            (z₁,z₂) = zvu2.data
            β₁ = vload(ptrβ, (n,)); γ₁ = vload(ptrγ, (n,));
            β₂ = vload(ptrβ, (vadd(W, n),), mask); γ₂ = vload(ptrγ, (vadd(W, n),), mask);
            vstore!(ptrx, z₁ * γ₁ + β₁, (n,));
            vstore!(ptrx, z₂ * γ₂ + β₂, (vadd(W, n),), mask);
        elseif scalar_less(n, N)
            state, zvu1 = f(state, Val{1}(), T)
            (z₁,) = zvu1.data
            β₁ = vload(ptrβ, (n,), mask); γ₁ = vload(ptrγ, (n,), mask);
            vstore!(ptrx, z₁ * γ₁ + β₁, (n,), mask);
        end
        storestate!(rng, state)
    end # GC preserve
    # @show state getstate(rng, Val{2}(), Val{W64}())
    # @assert state === getstate(rng, Val{2}(), Val{W64}())
    x
end
function random_sample_u4!(f::F, rng::AbstractVRNG{P}, x::AbstractArray{T}, ::StaticInt{0}, β, γ) where {F,P,T}
    state = getstate(rng, Val{P}(), Val{W64}())
    GC.@preserve x begin
        ptrx = zero_offsets(stridedpointer(x)); ptrβ = zero_offsets(stridedpointer(β)); ptrγ = zero_offsets(stridedpointer(γ));
        W = VectorizationBase.pick_vector_width(T); W2 = W+W; W3 = W2+W; W4 = W2+W2;
        Wval = VectorizationBase.pick_vector_width_val(T)
        N = length(x)
        n = MM(Wval, 0)
        while scalar_less(n, vadd(N, 1 - 4W))
            state, zvu4 = f(state, Val{4}(), T)
            (z₁,z₂,z₃,z₄) = zvu4.data
            β₁ = vload(ptrβ, (n,)); γ₁ = vload(ptrγ, (n,));
            β₂ = vload(ptrβ, (vadd(W, n),)); γ₂ = vload(ptrγ, (vadd(W, n),));
            β₃ = vload(ptrβ, (vadd(W2, n),)); γ₃ = vload(ptrγ, (vadd(W2, n),));
            β₄ = vload(ptrβ, (vadd(W3, n),)); γ₄ = vload(ptrγ, (vadd(W3, n),));
            vstore!(ptrx, z₁ * γ₁ + β₁, (n,));
            vstore!(ptrx, z₂ * γ₂ + β₂, (vadd(W,n),));
            vstore!(ptrx, z₃ * γ₃ + β₃, (vadd(W2,n),));
            vstore!(ptrx, z₄ * γ₄ + β₄, (vadd(W3,n),));
            n = vadd(W4, n) 
        end
        mask = VectorizationBase.mask(Wval, N)
        if scalar_less(n, vsub(N, 3W))
            state, zvu4 = f(state, Val{4}(), T)
            (z₁,z₂,z₃,z₄) = zvu4.data
            β₁ = vload(ptrβ, (n,)); γ₁ = vload(ptrγ, (n,));
            β₂ = vload(ptrβ, (vadd(W, n),)); γ₂ = vload(ptrγ, (vadd(W, n),));
            β₃ = vload(ptrβ, (vadd(W2, n),)); γ₃ = vload(ptrγ, (vadd(W2, n),));
            β₄ = vload(ptrβ, (vadd(W3, n),), mask); γ₄ = vload(ptrγ, (vadd(W3, n),), mask);
            vstore!(ptrx, z₁ * γ₁ + β₁, (n,));
            vstore!(ptrx, z₂ * γ₂ + β₂, (vadd(W,n),));
            vstore!(ptrx, z₃ * γ₃ + β₃, (vadd(W2,n),));
            vstore!(ptrx, z₄ * γ₄ + β₄, (vadd(W3,n),), mask);
        elseif scalar_less(n, vsub(N, 2W))
            state, zvu3 = f(state, Val{3}(), T)
            (z₁,z₂,z₃) = zvu3.data
            β₁ = vload(ptrβ, (n,)); γ₁ = vload(ptrγ, (n,));
            β₂ = vload(ptrβ, (vadd(W, n),)); γ₂ = vload(ptrγ, (vadd(W, n),));
            β₃ = vload(ptrβ, (vadd(W2, n),), mask); γ₃ = vload(ptrγ, (vadd(W2, n),), mask);
            vstore!(ptrx, z₁ * γ₁ + β₁, (n,));
            vstore!(ptrx, z₂ * γ₂ + β₂, (vadd(W,n),));
            vstore!(ptrx, z₃ * γ₃ + β₃, (vadd(W2,n),), mask);
        elseif scalar_less(n, vsub(N, W))
            state, zvu2 = f(state, Val{2}(), T)
            (z₁,z₂) = zvu2.data
            β₁ = vload(ptrβ, (n,)); γ₁ = vload(ptrγ, (n,));
            β₂ = vload(ptrβ, (vadd(W, n),), mask); γ₂ = vload(ptrγ, (vadd(W, n),), mask);
            vstore!(ptrx, z₁ * γ₁ + β₁, (n,));
            vstore!(ptrx, z₂ * γ₂ + β₂, (vadd(W,n),), mask);
        elseif scalar_less(n, N)
            state, zvu1 = f(state, Val{1}(), T)
            (z₁,) = zvu1.data
            β₁ = vload(ptrβ, (n,), mask); γ₁ = vload(ptrγ, (n,), mask);
            vstore!(ptrx, z₁ * γ₁ + β₁, (n,), mask);
        end        
        storestate!(rng, state)
    end # GC preserve
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

Random.rand(rng::AbstractVRNG, d1::Integer, dims::Vararg{Integer,N}) where {N} = rand!(rng, Array{Float64}(undef, d1, dims...))
Random.randn(rng::AbstractVRNG, d1::Integer, dims::Vararg{Integer,N}) where {N} = randn!(rng, Array{Float64}(undef, d1, dims...))
# Random.randexp(rng::AbstractVRNG, d1::Integer, dims::Vararg{Integer,N}) where {N} = randexp!(rng, Array{Float64}(undef, d1, dims...))


struct Buffer256{T} <: DenseVector{T}
    ptr::Ptr{T}
end
Base.pointer(b::Buffer256) = b.ptr
Base.length(::Buffer256) = 256
Base.size(::Buffer256) = (256,)
Base.getindex(b::Buffer256, i::Int) = vload(stridedpointer(b), (i-1,))
Base.strides(::Buffer256) = (1,)
VectorizationBase.ArrayInterface.contiguous_axis(::Type{<:Buffer256}) = VectorizationBase.ArrayInterface.Contiguous{1}()
VectorizationBase.ArrayInterface.contiguous_batch_size(::Type{<:Buffer256}) = VectorizationBase.ArrayInterface.ContiguousBatch{0}()
VectorizationBase.ArrayInterface.stride_rank(::Type{<:Buffer256}) = VectorizationBase.ArrayInterface.StrideRank{(1,)}()


function Random.rand(rng::AbstractVRNG)
    i = getrand64counter(rng)
    b = randbuffer64(rng)
    setrand64counter!(rng, i + 0x01)
    iszero(i) && rand!(rng, b)
    vload(pointer(b), VectorizationBase.LazyMulAdd{8,0}(i))
end
function Random.randn(rng::AbstractVRNG)
    i = getrandn64counter(rng)
    b = randnbuffer64(rng)
    setrandn64counter!(rng, i + 0x01)
    iszero(i) && randn!(rng, b)
    vload(pointer(b), VectorizationBase.LazyMulAdd{8,0}(i))
end

