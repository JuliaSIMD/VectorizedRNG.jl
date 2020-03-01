

# @enum PCG_Algorithm RXS_M_XS XSH_RR

# function default_pcg_type(W, T)
    # if W > VectorizationBase.pick_vector_width(T)
        # return RXS_M_XS
    # else
        # return XSH_RR
    # end
# end

abstract type  AbstractPCG{N} <: AbstractVRNG{N} end

mutable struct PCG{N} <: AbstractPCG{N}
    state::NTuple{N,Vec{W64,UInt64}}
    multiplier::NTuple{N,Vec{W64,UInt64}}
    PCG{N}(::UndefInitializer) where {N} = new{N}()
    function PCG(
        state::NTuple{N,Vec{W64,UInt64}},
        multiplier::NTuple{N,Vec{W64,UInt64}},
        increment::UInt64 = one(UInt64)
    ) where {N}
        pcg = PCG{N}(undef)
        pcg.state = state
        pcg.multiplier = multiplier
        pcg
    end        
end
struct PtrPCG{N} <: AbstractPCG{N}
    ptr::Ptr{UInt64}
end

struct PCGState{N,W} <: AbstractState{N,W}
    state::NTuple{N,Vec{W,UInt64}}
    multipliers::NTuple{N,Vec{W,UInt64}}
end

@inline Base.pointer(rng::AbstractPCG) = Base.unsafe_convert(Ptr{UInt64}, pointer_from_objref(rng))
@inline Base.pointer(rng::PtrPCG) = rng.ptr
# @inline Base.pointer(rng::AbstractPCG) = Base.unsafe_convert(Ptr{Vec{W64,UInt64}}, pointer_from_objref(rng))

@generated function PCG{N}(offset = 0) where N
    quote
        PCG(
            (Base.Cartesian.@ntuple $N n -> (Base.Cartesian.@ntuple $W64 w -> Core.VecElement(rand(UInt64)))),
            (Base.Cartesian.@ntuple $N n -> MULTIPLIERS[(Base.Threads.atomic_add!(MULT_NUMBER, 1) + offset - 1) % $(length(MULTIPLIERS)) + 1])
        )
    end
end
@generated function random_init_pcg!(pcg::AbstractPCG{N}, offset = 0) where {N}
    q = quote ptr = pointer(pcg) end
    for n ∈ 0:N-1
        n_quote = quote
            # state
            SIMDPirates.vstorea!(gep(ptr, $n * $W64), (Base.Cartesian.@ntuple $W64 w -> Core.VecElement(rand(UInt64))))
            # multiplier
            SIMDPirates.vstorea!(gep(ptr, $(N + n) * $W64), MULTIPLIERS[(Base.Threads.atomic_add!(MULT_NUMBER, 1) + offset * $N - 1) % $(length(MULTIPLIERS)) + 1])
        end
        push!(q.args, n_quote)
    end
    # push!(q.args, :(VectorizationBase.vstore!(gep(ptr, $(2N)*$W64), one(UInt64) + 2 * ((MULT_NUMBER[] + offset * N - 1) ÷ $(length(MULTIPLIERS))))))
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
            (Base.Cartesian.@ntuple $N n -> MULTIPLIERS[(Base.Threads.atomic_add!(MULT_NUMBER, 1) + offset - 1) % $(length(MULTIPLIERS)) + 1])
            # one(UInt64) + 2 * ((MULT_NUMBER[] + offset - 1) ÷ $(length(MULTIPLIERS)))
        )
    end
end

# @noinline function adjust_vector_width(W, T)
#     if T == XSH_RR
#         W >>>= 1
#     elseif T == RXS_M_XS
#         W = W
#     else
#         throw("Have not yet added support for $T.")
#     end
#     W
# end
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

@generated function getstate(rng::AbstractPCG{P}, ::Val{N}, ::Val{WV}) where {P,N,WV}
    @assert 8WV ≤ REGISTER_SIZE
    loads = min(N,P)
    q = quote
        # $(Expr(:meta,:inline))
        prng = pointer(rng)
        # increment = vone(Vec{$WV,UInt64})# vbroadcast(, prng + $(2P) * $REGISTER_SIZE)
    end
    states = Expr(:tuple)
    multipliers = Expr(:tuple)
    for n ∈ 1:loads
        state = Symbol(:state_, n)
        multiplier = Symbol(:multiplier_, n)
        push!(states.args, state); push!(multipliers.args, multiplier)
        push!(q.args, quote
              $state = vloada(Vec{$WV,UInt64}, gep(prng + $(W64 * (n-1))))
              $multiplier = vloada(Vec{$WV,UInt64}, gep(prng + $(W64 * (P + n-1))))
              end)
    end
    push!(q.args, Expr(:call, :PCGState, states, multipliers))
    q
end
@inline function getstate(rng::AbstractPCG{4}, ::Val{1}, ::Val{WV}) where {WV}
    prng = pointer(rng)
    states = (
        vloada(Vec{WV,UInt64}, prng),
    )
    multipliers = (
        vloada(Vec{WV,UInt64}, prng, 4W64),
    )
    PCGState(states, multipliers)
end
@inline function getstate(rng::AbstractPCG{4}, ::Val{2}, ::Val{WV}) where {WV}
    prng = pointer(rng)
    states = (
        vloada(Vec{WV,UInt64}, prng),
        vloada(Vec{WV,UInt64}, prng, W64)
    )
    multipliers = (
        vloada(Vec{WV,UInt64}, prng, 4W64),
        vloada(Vec{WV,UInt64}, prng, 5W64)
    )
    PCGState(states, multipliers)
end
@inline function getstate(rng::AbstractPCG{4}, ::Val{3}, ::Val{WV}) where {WV}
    prng = pointer(rng)
    states = (
        vloada(Vec{WV,UInt64}, prng),
        vloada(Vec{WV,UInt64}, prng, W64),
        vloada(Vec{WV,UInt64}, prng, 2W64)
    )
    multipliers = (
        vloada(Vec{WV,UInt64}, prng, 4W64),
        vloada(Vec{WV,UInt64}, prng, 5W64),
        vloada(Vec{WV,UInt64}, prng, 6W64)
    )
    PCGState(states, multipliers)
end
@inline function getstate(rng::AbstractPCG{4}, ::Val{4}, ::Val{WV}) where {WV}
    prng = pointer(rng)
    states = (
        vloada(Vec{WV,UInt64}, prng),
        vloada(Vec{WV,UInt64}, prng, W64),
        vloada(Vec{WV,UInt64}, prng, 2W64),
        vloada(Vec{WV,UInt64}, prng, 3W64)
    )
    multipliers = (
        vloada(Vec{WV,UInt64}, prng, 4W64),
        vloada(Vec{WV,UInt64}, prng, 5W64),
        vloada(Vec{WV,UInt64}, prng, 6W64),
        vloada(Vec{WV,UInt64}, prng, 7W64)
    )
    PCGState(states, multipliers)
end
@inline function storestate!(rng::AbstractPCG, s::PCGState{N,W}) where {N,W}
    q = quote
        $(Expr(:meta,:inline))
        prng = pointer(rng)
        states = s.state
    end
    for n ∈ 1:N
        push!(q.args, :(vstorea!(prng + $(REGISTER_SIZE * (n-1)), @inbounds($(Expr(:ref, :states, n))))))
    end
    push!(q.args, nothing)
    q
end
@inline function storestate!(rng::AbstractPCG, s::PCGState{2})
    prng = pointer(rng)
    states = s.state
    vstorea!(prng, (@inbounds states[1]))
    vstorea!(prng, (@inbounds states[2]), W64)
    nothing
end
@inline function storestate!(rng::AbstractPCG, s::PCGState{4})
    prng = pointer(rng)
    states = s.state
    vstorea!(prng, (@inbounds states[1]))
    vstorea!(prng, (@inbounds states[2]), W64)
    vstorea!(prng, (@inbounds states[3]), 2W64)
    vstorea!(prng, (@inbounds states[4]), 3W64)
    nothing
end


function append_n_rxs!(q, N, i = 0)
    for n ∈ 1:N
        state = Symbol(:state_, n)
        count = Symbol(:count_, n)
        push!(q.args, :($count = vuright_bitshift($state, Val{0x000000000000003b}())))
    end
    for n ∈ 1:N
        it = i + n
        state = Symbol(:state_, n)
        statenew = Symbol(:state_new_, n)
        push!(q.args, :($statenew = vmul($(Symbol(:multiplier_, n)), $state)))
    end
    for n ∈ 1:N
        count = Symbol(:count_, n)
        push!(q.args, :($count = vadd(five, $count)))
    end
    for n ∈ 1:N
        state = Symbol(:state_, n)
        count = Symbol(:count_, n)
        push!(q.args, :($count = vuright_bitshift($state, $count)))
    end
    for n ∈ 1:N
        state = Symbol(:state_, n)
        count = Symbol(:count_, n)
        push!(q.args, :($count = vxor($count, $state)))
    end
    for n ∈ 1:N
        it = i + n
        state = Symbol(:state_, n)
        statenew = Symbol(:state_new_, n)
        push!(q.args, :($state = vadd($statenew, 0x0000000000000001)))
    end
    for n ∈ 1:N
        xorshifted = Symbol(:xorshifted_, n)
        count = Symbol(:count_, n)
        push!(q.args, :($xorshifted = vmul($count, constmul)))
    end
    for n ∈ 1:N
        xorshifted = Symbol(:xorshifted_, n)
        xorshifted43 = Symbol(:xorshifted43_, n)
        push!(q.args, :($xorshifted43 = vuright_bitshift($xorshifted, Val{0x000000000000002b}())))
    end
    for n ∈ 1:N
        xorshifted = Symbol(:xorshifted_, n)
        xorshifted43 = Symbol(:xorshifted43_, n)
        out = Symbol(:out_, i + n)
        push!(q.args, :($out = vxor($xorshifted, $xorshifted43)))
    end
end
function rand_pcgPCG_RXS_M_XS_int64_quote(N, WV, Nreps)
    q = Expr(
        :block,
        Expr(:(=), :five, :(vbroadcast(Vec{$WV,UInt64}, 0x0000000000000005))),
        Expr(:(=), :constmul, :(vbroadcast(Vec{$WV,UInt64}, 0xaef17502108ef2d9)))
    )
    if Nreps > N
        NNrep, rr = divrem(Nreps, N)
        i = 0
        for nr ∈ 1:NNrep
            append_n_rxs!(q, N, i)
            i += N
        end
        for n ∈ 1:rr
            append_n_rxs!(q, rr, i)
        end
    else # Nreps <= N
        append_n_rxs!(q, Nreps, 0)
    end
    push!(q.args, nothing)
    q
end
# @noinline function rand_pcgPCG_XSH_RR_int32_quote(N, WV, Nreps)
#     WV32 = 2WV
#     q = quote end
#     if Nreps > N
#         NNrep, rr = divrem(Nreps, N)
#         i = 0
#         for nr ∈ 1:NNrep
#             for n ∈ 1:N
#                 i += 1
#                 state = Symbol(:state_, n)
#                 xorshifted = Symbol(:xorshifted_, i)
#                 rot = Symbol(:rot_, i)
#                 out = Symbol(:out_,i)
#                 push!(q.args, quote
#                       $xorshifted = vreinterpret(Vec{$WV32,UInt32}, vuright_bitshift(
#                         vxor(
#                             vuright_bitshift($state, vbroadcast(Vec{$WV,UInt64}, 0x0000000000000012)), $state
#                         ), vbroadcast(Vec{$WV,UInt64}, 0x000000000000001b)
#                       ))
#                       $rot = vreinterpret(Vec{$WV32,UInt32},vuright_bitshift($state, vbroadcast(Vec{$WV,UInt64}, 0x000000000000003b)))
#                       $state = vmuladd($(Symbol(:multiplier_, n)), $state, 0x0000000000000001)
# #                      $out = rotate($xorshifted, $rot)
#                       $out = vreinterpret(Vec{$(WV>>>1),UInt64}, extract_data(rotate(
#                           SVec(shufflevector($xorshifted, $(Expr(:call,Expr(:curly,:Val,Expr(:tuple, [w for w ∈ 0:2:WV32-1]...)))))),
#                           SVec(shufflevector($rot, $(Expr(:call,Expr(:curly,:Val,Expr(:tuple,[w for w ∈ 0:2:WV32-1]...))))))
#                       )))
#                       end)
#             end
#         end
#         for n ∈ 1:rr
#             i += 1
#             state = Symbol(:state_, n)
#             xorshifted = Symbol(:xorshifted_, i)
#             rot = Symbol(:rot_, i)
#             out = Symbol(:out_,i)
#             push!(q.args, quote
#                 $xorshifted = vreinterpret(Vec{$WV32,UInt32}, vuright_bitshift(
#                     vxor(
#                         vuright_bitshift($state, vbroadcast(Vec{$WV,UInt}, 0x0000000000000012)), $state
#                     ), vbroadcast(Vec{$WV,UInt}, 0x000000000000001b)
#                 ))
#                 $rot = vreinterpret(Vec{$WV32,UInt32},vuright_bitshift($state, vbroadcast(Vec{$WV,UInt64}, 0x000000000000003b)))
#                 $state = vmuladd($(Symbol(:multiplier_, n)), $state, 0x0000000000000001)
#                   $out = vreinterpret(Vec{$(WV>>>1),UInt64},extract_data(rotate(
#                       SVec(shufflevector($xorshifted, $(Expr(:call,Expr(:curly,:Val,Expr(:tuple,[w for w ∈ 0:2:WV32-1]...)))))),
#                       SVec(shufflevector($rot, $(Expr(:call,Expr(:curly,:Val,Expr(:tuple,[w for w ∈ 0:2:WV32-1]...))))))
#                   )))
# #                $out = rotate($xorshifted, $rot)
#             end)
#         end
#     else # Nreps <= N
#         for n ∈ 1:Nreps
#             state = Symbol(:state_, n)
#             xorshifted = Symbol(:xorshifted_, n)
#             rot = Symbol(:rot_, n)
#             out = Symbol(:out_, n)
#             mult = Symbol(:multiplier_,n)
#             push!(q.args, quote
#                   $xorshifted = vreinterpret(Vec{$WV32,UInt32}, vuright_bitshift(
#                     vxor(
#                         vuright_bitshift($state, vbroadcast(Vec{$WV,UInt}, 0x0000000000000012)), $state
#                     ), vbroadcast(Vec{$WV,UInt}, 0x000000000000001b)
#                 ))
#                 $rot = vreinterpret(Vec{$WV32,UInt32},vuright_bitshift($state, vbroadcast(Vec{$WV,UInt64}, 0x000000000000003b)))
#                   $state = vmuladd($mult, $state, 0x0000000000000001)
#                   $out = vreinterpret(Vec{$(WV>>>1),UInt64},extract_data(rotate(
#                       SVec(shufflevector($xorshifted, $(Expr(:call,Expr(:curly,:Val,Expr(:tuple,[w for w ∈ 0:2:WV32-1]...)))))),
#                       SVec(shufflevector($rot, $(Expr(:call,Expr(:curly,:Val,Expr(:tuple,[w for w ∈ 0:2:WV32-1]...))))))
#                   )))
#                   end)
#         end
#     end
#     push!(q.args, nothing)
#     q
# end





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

# @generated function random_xsh_rr(state::NTuple{P,Vec{W,UInt64}}, multiplier::NTuple{P,Vec{W,UInt64}}, ::Val{N}) where {N,P,W}
#     quote
#         $(Expr(:meta,:inline))
#         $(state_tup_expr(P)) = state
#         $(multiplier_tup_expr(P)) = multiplier
#         $(rand_pcgPCG_XSH_RR_int32_quote(P, W, N))
#         state = $(state_tup_expr(P))
#         out = $(out_tup_expr(N))
#         state, out
#     end    
# end
@generated function nextstate(s::PCGState{P,W}, ::Val{N}) where {N,P,W}
    quote
        $(Expr(:meta,:inline))
        state = s.state
        multiplier = s.multipliers
        $(state_tup_expr(P)) = state
        $(multiplier_tup_expr(P)) = multiplier
        $(rand_pcgPCG_RXS_M_XS_int64_quote(P, W, N))
        state = $(state_tup_expr(P))
        out = $(out_tup_expr(N))
        PCGState(state, multiplier), out
    end    
end

# @generated Val2(::Val{N}) where {N} = Val{N<<1}()
# @generated ValH(::Val{N}) where {N} = Val{(N+1)>>>1}()







Random.rand(pcg::AbstractPCG, ::Type{UInt32}) = Base.unsafe_trunc(UInt32, rand(pcg, UInt64))
Random.rand(pcg::AbstractPCG, ::Type{Int64}) = reinterpret(Int64, rand(pcg, UInt64))
Random.rand(pcg::AbstractPCG, ::Type{Int32}) = reinterpret(Int32, rand(pcg, UInt32))
Random.rand(pcg::AbstractPCG, ::Type{T} = Float64) where {T} = @inbounds rand(pcg,Vec{1,T})[1].value
Random.rng_native_52(::AbstractPCG) = UInt64




