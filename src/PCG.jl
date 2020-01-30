

@enum PCG_Algorithm RXS_M_XS XSH_RR

function default_pcg_type(W, T)
    if W > VectorizationBase.pick_vector_width(T)
        return RXS_M_XS
    else
        return XSH_RR
    end
end

abstract type  AbstractPCG{N} <: Random.AbstractRNG end

mutable struct PCG{N} <: AbstractPCG{N}
    state::NTuple{N,Vec{W64,UInt64}}
    multiplier::NTuple{N,Vec{W64,UInt64}}
    increment::UInt64
    PCG{N}(::UndefInitializer) where {N} = new{N}()
    function PCG(
        state::NTuple{N,Vec{W64,UInt64}},
        multiplier::NTuple{N,Vec{W64,UInt64}},
        increment::UInt64 = one(UInt64)
    ) where {N}
        pcg = PCG{N}(undef)
        pcg.state = state
        pcg.multiplier = multiplier
        pcg.increment = increment
        pcg
    end        
end
struct PtrPCG{N} <: AbstractPCG{N}
    ptr::Ptr{UInt64}
end

@inline Base.pointer(rng::AbstractPCG) = Base.unsafe_convert(Ptr{UInt64}, pointer_from_objref(rng))
@inline Base.pointer(rng::PtrPCG) = rng.ptr
# @inline Base.pointer(rng::AbstractPCG) = Base.unsafe_convert(Ptr{Vec{W64,UInt64}}, pointer_from_objref(rng))

@generated function PCG{N}(offset = 0) where N
    quote
        PCG(
            (Base.Cartesian.@ntuple $N n -> (Base.Cartesian.@ntuple $W64 w -> Core.VecElement(rand(UInt64)))),
            (Base.Cartesian.@ntuple $N n -> MULTIPLIERS[(Base.Threads.atomic_add!(MULT_NUMBER, 1) + offset - 1) % $(length(MULTIPLIERS)) + 1]),
            one(UInt64) + 2 * ((MULT_NUMBER[] + offset - 1) ÷ $(length(MULTIPLIERS)))
        )
    end
end


@generated function random_init_pcg!(pcg::AbstractPCG{N}, offset = 0) where {N}
    q = quote ptr = pointer(pcg) end
    for n ∈ 0:N-1
        n_quote = quote
            # state
            SIMDPirates.vstore!(ptr + $n * $REGISTER_SIZE, Base.Cartesian.@ntuple $W64 w -> Core.VecElement(rand(UInt64)))
            # multiplier
            SIMDPirates.vstore!(ptr + $(N + n) * $REGISTER_SIZE, MULTIPLIERS[(Base.Threads.atomic_add!(MULT_NUMBER, 1) + offset * $N - 1) % $(length(MULTIPLIERS)) + 1])
        end
        push!(q.args, n_quote)
    end
    push!(q.args, :(VectorizationBase.store!(ptr + $(2N)*$REGISTER_SIZE, one(UInt64) + 2 * ((MULT_NUMBER[] + offset * N - 1) ÷ $(length(MULTIPLIERS))) )))
    push!(q.args, :pcg)
    q
end

@generated function PCG(seeds::NTuple{WN,UInt64}, offset = 0) where WN
    W, Wshift = VectorizationBase.pick_vector_width_shift(Float64)
    Wm1 = W - 1
    N = WN >>> Wshift
    @assert WN & Wm1 == 0
    quote
        PCG(
            (Base.Cartesian.@ntuple $N n -> (Base.Cartesian.@ntuple $W64 w -> Core.VecElement(seeds[w + $W * (n-1)]))),
            (Base.Cartesian.@ntuple $N n -> MULTIPLIERS[(Base.Threads.atomic_add!(MULT_NUMBER, 1) + offset - 1) % $(length(MULTIPLIERS)) + 1]),
            one(UInt64) + 2 * ((MULT_NUMBER[] + offset - 1) ÷ $(length(MULTIPLIERS)))
        )
    end
end

@noinline function adjust_vector_width(W, T)
    if T == XSH_RR
        W >>>= 1
    elseif T == RXS_M_XS
        W = W
    else
        throw("Have not yet added support for $T.")
    end
    W
end
function name_n_tup_expr(name, N)
    tup = Expr(:tuple)
    for n ∈ 1:N
        name_n = Symbol(name, n)
        push!(tup.args, name_n);
    end
    tup
end
state_tup_expr(N) = name_n_tup_expr(:state_, N)
multiplier_tup_expr(N) = name_n_tup_expr(:multiplier_, N)
out_tup_expr(N) = name_n_tup_expr(:out_, N)

@generated function load_vectors(rng::AbstractPCG{P}, ::Val{N}, ::Val{WV}) where {P,N,WV}
    @assert 8WV ≤ REGISTER_SIZE
    loads = min(N,P)
    q = quote
        prng = pointer(rng)
        increment = vbroadcast(Vec{$WV,UInt64}, prng + $(2P) * $REGISTER_SIZE)
    end
    states = Expr(:tuple)
    multipliers = Expr(:tuple)
    for n ∈ 1:loads
        state = Symbol(:state_, n)
        multiplier = Symbol(:multiplier_, n)
        push!(states.args, state); push!(multipliers.args, multiplier)
        push!(q.args, quote
              $state = vload(Vec{$WV,UInt64}, prng + $(REGISTER_SIZE * (n-1)) )
              $multiplier = vload(Vec{$WV,UInt64}, prng + $(REGISTER_SIZE * (P + n-1)) )
              end)
    end
    push!(q.args, Expr(:tuple, states, multipliers, :increment))
    q
end
@generated function store_state!(rng::AbstractPCG, states::NTuple{N,Vec{W,T}}) where {N,W,T}
    q = quote
        prng = pointer(rng)
    end
    for n ∈ 1:N
        push!(q.args, :(vstore!(prng + $(REGISTER_SIZE * (n-1)), @inbounds($(Expr(:ref, :states, n))))))
    end
    push!(q.args, nothing)
    q
end

@noinline function rand_pcgPCG_RXS_M_XS_int64_quote(N, WV, Nreps)
    q = quote end
    if Nreps > N
        NNrep, rr = divrem(Nreps, N)
        i = 0
        for nr ∈ 1:NNrep
            for n ∈ 1:N
                i += 1
                state = Symbol(:state_, n)
                xorshifted = Symbol(:xorshifted_, i)
                count = Symbol(:count_, i)
                out = Symbol(:out_, i)
                push!(q.args, quote
                    $count = vadd(vbroadcast(Vec{$WV,UInt64}, UInt64(5)), vuright_bitshift($state, Val{59}()))
                    $xorshifted = vmul(vxor(
                            vuright_bitshift($state, $count), $state
                        ), 0xaef17502108ef2d9)
                    $state = vmuladd($(Symbol(:multiplier_, n)), $state, increment)
                    $out = vxor($xorshifted, vuright_bitshift($xorshifted, Val{43}()))
                end)
            end
        end
        for n ∈ 1:rr
            i += 1
            state = Symbol(:state_, n)
            xorshifted = Symbol(:xorshifted_, i)
            count = Symbol(:count_, i)
            out = Symbol(:out_, i)
            push!(q.args, quote
                $count = vadd(vbroadcast(Vec{$WV,UInt64}, UInt64(5)), vuright_bitshift($state, Val{59}()))
                $xorshifted = vmul(vxor(
                        vuright_bitshift($state, $count), $state
                    ), 0xaef17502108ef2d9)
                $state = vmuladd($(Symbol(:multiplier_, n)), $state, increment)
                $out = vxor($xorshifted, vuright_bitshift($xorshifted, Val{43}()))
            end)
        end
    else # Nreps <= N
        for n ∈ 1:Nreps
            state = Symbol(:state_, n)
            xorshifted = Symbol(:xorshifted_, n)
            count = Symbol(:count_, n)
            out = Symbol(:out_, n)
            mult = Symbol(:multiplier_, n)
            push!(q.args, quote
                $count = vadd(vbroadcast(Vec{$WV,UInt64}, UInt64(5)), vuright_bitshift($state, Val{59}()))
                $xorshifted = vmul(vxor(
                        vuright_bitshift($state, $count), $state
                    ), 0xaef17502108ef2d9)
                $state = vmuladd($mult, $state, increment)
                $out = vxor($xorshifted, vuright_bitshift($xorshifted, Val{43}()))
                  end)
        end
    end
    push!(q.args, nothing)
    q
end

@inline rotate(x, r) = x >>> r | x << (-r & 31)
@inline function rotate(x::Vec{W,T1}, r::Vec{W,T2}) where {W,T1,T2}
    xshiftright = SIMDPirates.vuright_bitshift(x, r)
    nra31 = SIMDPirates.vand(SIMDPirates.vsub(r), SIMDPirates.vbroadcast(Vec{W,T2}, T2(31)))
    xshiftleft = SIMDPirates.vleft_bitshift(x, nra31)
    SIMDPirates.vor(xshiftright, xshiftleft)
end

@noinline function rand_pcgPCG_XSH_RR_int32_quote(N, WV, Nreps)
    WV32 = 2WV
    q = quote end
    if Nreps > N
        NNrep, rr = divrem(Nreps, N)
        i = 0
        for nr ∈ 1:NNrep
            for n ∈ 1:N
                i += 1
                state = Symbol(:state_, n)
                xorshifted = Symbol(:xorshifted_, i)
                rot = Symbol(:rot_, i)
                out = Symbol(:out_,i)
                push!(q.args, quote
                      $xorshifted = vreinterpret(Vec{$WV32,UInt32}, vuright_bitshift(
                        vxor(
                            vuright_bitshift($state, vbroadcast(Vec{$WV,UInt64}, 18)), $state
                        ), vbroadcast(Vec{$WV,UInt64}, 27)
                      ))
                      $rot = vreinterpret(Vec{$WV32,UInt32},vuright_bitshift($state, vbroadcast(Vec{$WV,UInt64}, 59)))
                      $state = vmuladd($(Symbol(:multiplier_, n)), $state, increment)
#                      $out = rotate($xorshifted, $rot)
                      $out = vreinterpret(Vec{$(WV>>>1),UInt64}, extract_data(rotate(
                          SVec(shufflevector($xorshifted, $(Expr(:call,Expr(:curly,:Val,Expr(:tuple, [w for w ∈ 0:2:WV32-1]...)))))),
                          SVec(shufflevector($rot, $(Expr(:call,Expr(:curly,:Val,Expr(:tuple,[w for w ∈ 0:2:WV32-1]...))))))
                      )))
                      end)
            end
        end
        for n ∈ 1:rr
            i += 1
            state = Symbol(:state_, n)
            xorshifted = Symbol(:xorshifted_, i)
            rot = Symbol(:rot_, i)
            out = Symbol(:out_,i)
            push!(q.args, quote
                $xorshifted = vreinterpret(Vec{$WV32,UInt32}, vuright_bitshift(
                    vxor(
                        vuright_bitshift($state, vbroadcast(Vec{$WV,UInt}, 18)), $state
                    ), vbroadcast(Vec{$WV,UInt}, 27)
                ))
                $rot = vreinterpret(Vec{$WV32,UInt32},vuright_bitshift($state, vbroadcast(Vec{$WV,UInt64}, 59)))
                $state = vmuladd($(Symbol(:multiplier_, n)), $state, increment)
                  $out = vreinterpret(Vec{$(WV>>>1),UInt64},extract_data(rotate(
                      SVec(shufflevector($xorshifted, $(Expr(:call,Expr(:curly,:Val,Expr(:tuple,[w for w ∈ 0:2:WV32-1]...)))))),
                      SVec(shufflevector($rot, $(Expr(:call,Expr(:curly,:Val,Expr(:tuple,[w for w ∈ 0:2:WV32-1]...))))))
                  )))
#                $out = rotate($xorshifted, $rot)
            end)
        end
    else # Nreps <= N
        for n ∈ 1:Nreps
            state = Symbol(:state_, n)
            xorshifted = Symbol(:xorshifted_, n)
            rot = Symbol(:rot_, n)
            out = Symbol(:out_, n)
            mult = Symbol(:multiplier_,n)
            push!(q.args, quote
                  $xorshifted = vreinterpret(Vec{$WV32,UInt32}, vuright_bitshift(
                    vxor(
                        vuright_bitshift($state, vbroadcast(Vec{$WV,UInt}, 18)), $state
                    ), vbroadcast(Vec{$WV,UInt}, 27)
                ))
                $rot = vreinterpret(Vec{$WV32,UInt32},vuright_bitshift($state, vbroadcast(Vec{$WV,UInt64}, 59)))
                  $state = vmuladd($mult, $state, increment)
                  $out = vreinterpret(Vec{$(WV>>>1),UInt64},extract_data(rotate(
                      SVec(shufflevector($xorshifted, $(Expr(:call,Expr(:curly,:Val,Expr(:tuple,[w for w ∈ 0:2:WV32-1]...)))))),
                      SVec(shufflevector($rot, $(Expr(:call,Expr(:curly,:Val,Expr(:tuple,[w for w ∈ 0:2:WV32-1]...))))))
                  )))
                  end)
        end
    end
    push!(q.args, nothing)
    q
end



MatchingUInt(::Type{Vec{W,Float64}}) where {W} = Vec{W,UInt64}
MatchingUInt(::Type{NTuple{N,Vec{W,Float64}}}) where {N,W} = NTuple{N,Vec{W,UInt64}}

@generated MatchingUInt(::Type{Vec{W,Float32}}) where {W} = Vec{cld(W,2),UInt64}
@generated MatchingUInt(::Type{NTuple{N,Vec{W,Float32}}}) where {N,W} = NTuple{N,Vec{cld(W,2),UInt64}}

@generated MatchingFloat32(::Type{Vec{W,UInt64}}) where {W} = Vec{W<<1,Float32}


"""
The masks mask off bits so that a set of bits will be within the range 1.0 and prevfloat(2.0).
They do this via first using the bitwise-and to set all the exponent bits and the sign-bit to 0,
and then using the bitwise-or to set all but the first exponent bit to 1.
Neither operation touches the fraction-bits.

For example, you can see the binary representation of a double precision number on wikipedia:
https://en.wikipedia.org/wiki/Double-precision_floating-point_format
and then
julia> bitstring(0x000fffffffffffff)
"0000000000001111111111111111111111111111111111111111111111111111"

julia> bitstring(0x3ff0000000000000)
"0011111111110000000000000000000000000000000000000000000000000000"

so that `(x & 0x000fffffffffffff) | 0x3ff0000000000000` has the desired effect of setting the sign
bit to 0, and the exponent bit to be within the proper range.
Note that the second string has 10 "1"s, so that it is 2^10.
For double precision, the exponential contribution is 2^(e - 1023). With e = 2^10,
we have 2^(2^10 - 1023) = 2^(1024 - 1023) = 2^1 = 2.
Thus, the fraction specifies where in the range between 1.0 and prevfloat(2.0).
If the fraction bits are completely random, the distribution will be uniform between 1.0 and prevfloat(2.0).

The reason we target that range (1.0 to prevfloat(2.0)) is to get a nice, uniform distribution.
We can get uniform(0, 1] via 2 - u, or a uniform[0,1) via u - 1.
Choosing other fractional intervals, eg [0.5, 1.0) or [2.0, 4.0) would take more work to translate.
Alternatively, if we tried to span multiple fractional intervals, suddenly trying to get a fairly
uniform distribution would get complicated.
"""
@inline mask(x, ::Type{Float64}) = reinterpret(Float64,(x & 0x000fffffffffffff) | 0x3ff0000000000000)
@inline mask(x, ::Type{Float32}) = reinterpret(Float32,(x & 0x007fffff) | 0x3f800000)
@inline mask(v::Vec{W,UInt64}, ::Type{Float64}) where {W} = vreinterpret(Vec{W,Float64}, vor(vand(v, 0x000fffffffffffff), 0x3ff0000000000000))
@inline mask(v::Vec{W,UInt64}, ::Type{Float32}) where {W} = vreinterpret(MatchingFloat32(Vec{W,UInt64}), vor(vand(v, 0x007fffff), 0x3f800000))

@generated function random_xsh_rr(state::NTuple{P,Vec{W,UInt64}}, multiplier::NTuple{P,Vec{W,UInt64}}, increment::Vec{W,UInt64}, ::Val{N}) where {N,P,W}
    quote
        $(Expr(:meta,:inline))
        $(state_tup_expr(P)) = state
        $(multiplier_tup_expr(P)) = multiplier
        $(rand_pcgPCG_XSH_RR_int32_quote(P, W, N))
        state = $(state_tup_expr(P))
        out = $(out_tup_expr(N))
        state, out
    end    
end
@generated function random_rxs_m_xs(state::NTuple{P,Vec{W,UInt64}}, multiplier::NTuple{P,Vec{W,UInt64}}, increment::Vec{W,UInt64}, ::Val{N}) where {N,P,W}
    quote
        $(Expr(:meta,:inline))
        $(state_tup_expr(P)) = state
        $(multiplier_tup_expr(P)) = multiplier
        $(rand_pcgPCG_RXS_M_XS_int64_quote(P, W, N))
        state = $(state_tup_expr(P))
        out = $(out_tup_expr(N))
        state, out
    end    
end
@generated Val2(::Val{N}) where {N} = Val{N<<1}()
@generated ValH(::Val{N}) where {N} = Val{(N+1)>>>1}()
@inline function Random.rand(rng::AbstractPCG{P}, ::Type{Vec{W,UInt32}}) where {P,W}
    state, mult, incr = load_vectors(rng, Val{1}(), ValH(Val{W}()))
    state, (out,) = random_xsh_rr(state, mult, incr, Val{1}())
    store_state!(rng, state)
    out
end
@inline function Random.rand(rng::AbstractPCG{P}, ::Type{Vec{W,UInt64}}) where {W,P}
    state, mult, incr = load_vectors(rng, Val{1}(), Val{W}())
    state, (out,) = random_rxs_m_xs(state, mult, incr, Val{1}())
    store_state!(rng, state)
    out
end
@inline function Random.rand(rng::AbstractPCG{P}, ::Type{NTuple{N,Vec{W,UInt32}}}) where {P,W,N}
    state, mult, incr = load_vectors(rng, Val{N}(), ValH(Val{W}()))
    state, out = random_xsh_rr(state, mult, incr, Val{N}())
    store_state!(rng, state)
    out
end
@inline function Random.rand(rng::AbstractPCG{P}, ::Type{NTuple{N,Vec{W,UInt64}}}) where {W,P,N}
    state, mult, incr = load_vectors(rng, Val{N}(), Val{W}())
    state, out = random_rxs_m_xs(state, mult, incr, Val{N}())
    store_state!(rng, state)
    out
end


@inline oneopenconst(::Type{Float64}) = 0.9999999999999999
@inline oneopenconst(::Type{Float32}) = 0.99999994f0

"""
Samples uniformly from (0.0,1.0)
"""
@inline function Random.rand(rng::AbstractPCG, ::Type{Vec{W,T}}) where {W,T}
    u = rand(rng, Vec{W,UInt64})
    vsub(mask(u, T), oneopenconst(T))
end
"""
if l < u,
Samples uniformly from [l,u)
else,
Samples uniformly from (u,l]

That is, the "l" side of the interval is closed, and the "u" side is open.
"""
@inline function Random.rand(rng::AbstractPCG, ::Type{Vec{W,T}}, l::T, u::T) where {W,T<:Union{Float32,Float64}}
    s = u - l
    b = l - s
    u = rand(rng, MatchingUInt(Vec{W,T}))
    vfmadd(s, mask(u, T), b)
end

@generated function Random.rand(rng::AbstractPCG, ::Type{NTuple{N,Vec{W,T}}}) where {N,W,T<:Union{Float32,Float64}}
    quote
        $(Expr(:meta,:inline))
        u = rand(rng, MatchingUInt(NTuple{N,Vec{W,T}}))
        $(Expr(:tuple, [:(vsub(mask(@inbounds(u[$n]), T), oneopenconst(T))) for n ∈ 1:N]...))
    end
end
@generated function Random.rand(rng::AbstractPCG, ::Type{NTuple{N,Vec{W,T}}}, l::T, u::T) where {N,W,T<:Union{Float32,Float64}}
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
    c = vcopysign(c, u2)
    l = log01(u2,T)
    # @show s, c, l
    r = vsqrt(vmul(-2.0,l))
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
    push!(q.args, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), ib))
    push!(q.args, Expr(:tuple, [Symbol(:n_,n) for n ∈ 0:N-1]...))
    q
end

@inline function Random.randn(rng::AbstractPCG, ::Type{NTuple{N,Vec{W,T}}}) where {N,W,T}
    u = rand(rng, MatchingUInt(NTuple{N,Vec{W,T}}))
    random_normal(u, T)
end

@inline function random_normal(state::NTuple{P,Vec{W,UInt64}}, mult::NTuple{P,Vec{W,UInt64}}, incr::Vec{W,UInt64}, ::Val{N}, ::Type{T}) where {P,N,W,T}
    state, u = random_rxs_m_xs(state, mult, incr, Val{N}())
    state, random_normal(u, T)
end

function random_sample!(f, x::AbstractArray{Float64}, rng::AbstractPCG{4})# where {P}
    state, mult, incr = load_vectors(rng, Val{4}(), Val{W64}())
    ptrx = pointer(x)
    N = length(x)
    n = 0
    @inbounds while n < N + 1 - 4W64
        state, (z₁,z₂,z₃,z₄) = f(state, mult, incr, Val{4}(), Float64)
        vstore!(ptrx + 8n, z₁); n += W64
        vstore!(ptrx + 8n, z₂); n += W64
        vstore!(ptrx + 8n, z₃); n += W64
        vstore!(ptrx + 8n, z₄); n += W64
    end
    mask = VectorizationBase.masktable(Val{W64}(), N & 7)
    if n < N + 1 - 3W64
        state, (z₁,z₂,z₃,z₄) = f(state, mult, incr, Val{4}(), Float64)
        vstore!(ptrx + 8n, z₁); n += W64
        vstore!(ptrx + 8n, z₂); n += W64
        vstore!(ptrx + 8n, z₃); n += W64
        vstore!(ptrx + 8n, z₄, mask);
    elseif n < N + 1 - 2W64
        state, (z₁,z₂,z₃) = f(state, mult, incr, Val{3}(), Float64)
        vstore!(ptrx + 8n, z₁); n += W64
        vstore!(ptrx + 8n, z₂); n += W64
        vstore!(ptrx + 8n, z₃, mask);
    elseif n < N + 1 - 1W64
        state, (z₁,z₂) = f(state, mult, incr, Val{2}(), Float64)
        vstore!(ptrx + 8n, z₁); n += W64
        vstore!(ptrx + 8n, z₂, mask);
    elseif n < N + 1
        state, (z₁,) = f(state, mult, incr, Val{1}(), Float64)
        vstore!(ptrx + 8n, z₁, mask);
    end        
    store_state!(rng, state)
    x
end
function Random.randn!(rng::AbstractPCG{4}, x::AbstractVector{Float64})
    random_sample!(random_normal, x, rng)
end

@noinline function rand_loop_quote(P, T, rngexpr, args...)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    statetup = Expr(:tuple, [Symbol(:state_,p) for p in 1:P]...)
    multtup = Expr(:tuple, [Symbol(:multiplier_,p) for p in 1:P]...)
    quote
        ptr_A = pointer(A)
        prng = pointer(rng)
        $([:($(Symbol(:state_, p)) = vload(Vec{$W64,UInt64}, prng + $(REGISTER_SIZE * (p-1)) )) for p in 1:P]...)
        $([:($(Symbol(:multiplier_, p)) = vload(Vec{$W64,UInt64}, prng + $(REGISTER_SIZE * (P + p-1)) )) for p in 1:P]...)
        increment = vbroadcast(Vec{$W64,UInt64}, prng + $(2P) * REGISTER_SIZE)
        L = length(A)
        nrep, nrem = divrem(L, $(P*W))
        GC.@preserve A begin
            for i ∈ 0:nrep-1
                r, $statetup = $rngexpr($statetup, $multtup, increment, NTuple{$P,Vec{$W,$T}}, $(args...))
                Base.Cartesian.@nexprs $P n -> begin
                    @inbounds SIMDPirates.vstore!(ptr_A + $(sizeof(T)*W) * ( (n-1) + $(P)*i ), r[n])
                end
            end
            if nrem > 0
                # r, $statetup = $rngexpr($statetup, $multtup, increment, NTuple{$P,Vec{$W,$T}}, $(args...))
                nremrep = nrem >>> $Wshift
                nremrem = nrem & $(W - 1)
                for n ∈ 1:nremrep
                    r_1, (state_1,) = $rngexpr((state_1,), (multiplier_1,), increment, Tuple{Vec{$W,$T}}, $(args...))
                    @inbounds SIMDPirates.vstore!(ptr_A + $(sizeof(T)*W) * ( (n-1) + $(P)*nrep ), r_1[1])
                end
                if nremrem > 0
                    r_1, (state_1,) = $rngexpr((state_1,), (multiplier_1,), increment, Tuple{Vec{$W,$T}}, $(args...))
                    @inbounds SIMDPirates.vstore!(ptr_A + $(sizeof(T)*W) * (nremrep + $(P)*nrep ), r_1[1], VectorizationBase.mask(T,nremrem))
                end
            end
            $([:(vstore!(prng + $(REGISTER_SIZE * (p-1)), $(Symbol(:state_,p)))) for p in 1:P]...)
            return A
        end
    end
end

@generated function Random.rand!(rng::AbstractPCG{P}, A::AbstractArray{T}, ::Val{PCG_TYPE} = Val{RXS_M_XS}()) where {T <: Real, P, PCG_TYPE}
    rand_loop_quote(adjust_vector_width(P, PCG_TYPE), T, :rand)
end
@generated function Random.rand!(rng::AbstractPCG{P}, A::AbstractArray{T}, l::T, u::T, ::Val{PCG_TYPE} = Val{RXS_M_XS}()) where {P,T <: Real, PCG_TYPE}
    rand_loop_quote(adjust_vector_width(P, PCG_TYPE), T, :rand, :l, :u)
end
@generated function Random.randexp!(rng::AbstractPCG{P}, A::AbstractArray{T}, ::Val{PCG_TYPE} = Val{RXS_M_XS}()) where {P,T <: Real, PCG_TYPE}
    rand_loop_quote(adjust_vector_width(P, PCG_TYPE), T, :randexp)
end
@generated function Random.randn!(
    rng::AbstractPCG{P}, A::AbstractArray{T}, ::Val{PCG_TYPE} = Val{RXS_M_XS}()
) where {P, T <: Real, PCG_TYPE}
#) where {T <: Real, P, PCG_TYPE}
    rand_loop_quote(adjust_vector_width(P, PCG_TYPE), T, :randn)
end
Random.rand(rng::AbstractPCG, d1::Integer, dims::Vararg{Integer,N} where N) = rand!(rng, Array{Float64}(undef, d1, dims...))
Random.randn(rng::AbstractPCG, d1::Integer, dims::Vararg{Integer,N} where N) = randn!(rng, Array{Float64}(undef, d1, dims...))
Random.randexp(rng::AbstractPCG, d1::Integer, dims::Vararg{Integer,N} where N) = randexp!(rng, Array{Float64}(undef, d1, dyims...))


Random.rand(pcg::AbstractPCG, ::Type{UInt32}) = Base.unsafe_trunc(UInt32, rand(pcg, UInt64))
Random.rand(pcg::AbstractPCG, ::Type{Int64}) = reinterpret(Int64, rand(pcg, UInt64))
Random.rand(pcg::AbstractPCG, ::Type{Int32}) = reinterpret(Int32, rand(pcg, UInt32))
Random.rand(pcg::AbstractPCG, ::Type{T} = Float64) where {T} = @inbounds rand(pcg,Vec{1,T})[1].value
Random.rng_native_52(::AbstractPCG) = UInt64




