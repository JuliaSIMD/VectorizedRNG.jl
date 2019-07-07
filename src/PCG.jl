

TypeVectorWidth(::Type{Float64}) = W64
TypeVectorWidth(::Type{Float32}) = W32
TypeVectorWidth(::Type{UInt64}) = W64
TypeVectorWidth(::Type{Int64}) = W64
TypeVectorWidth(::Type{UInt32}) = W32
TypeVectorWidth(::Type{Int32}) = W32
TypeVectorWidth(::Type{T}) where T = REGISTER_SIZE ÷ sizeof(T)

abstract type AbstractPCG_TYPE end

struct XSH_RR <: AbstractPCG_TYPE end
struct RXS_M_XS <: AbstractPCG_TYPE end

function default_pcg_type(W, T)
    if W > TypeVectorWidth(T)
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



@inline Base.pointer(rng::AbstractPCG) = Base.unsafe_convert(Ptr{UInt64}, pointer_from_objref(rng))
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


@generated function random_init_pcg!(pcg::PCG{N}, offset = 0) where {N}
    quote
        pcg.state = Base.Cartesian.@ntuple $N n -> (Base.Cartesian.@ntuple $W64 w -> Core.VecElement(rand(UInt64)))
        pcg.multiplier = Base.Cartesian.@ntuple $N n -> MULTIPLIERS[(Base.Threads.atomic_add!(MULT_NUMBER, 1) + offset * N - 1) % $(length(MULTIPLIERS)) + 1]
        pcg.increment = one(UInt64) + 2 * ((MULT_NUMBER[] + offset * N - 1) ÷ $(length(MULTIPLIERS)))
    end
end

@generated function PCG(seeds::NTuple{WN,UInt64}, offset = 0) where WN
    W, Wshift = VectorizationBase.pick_vector_width_shift(Float64)
    Wm1 = W - 1
    N = WN >> Wshift
    @assert WN & Wm1 == 0
    quote
        PCG(
            (Base.Cartesian.@ntuple $N n -> (Base.Cartesian.@ntuple $W64 w -> Core.VecElement(seeds[w + $W * (n-1)]))),
            (Base.Cartesian.@ntuple $N n -> MULTIPLIERS[(Base.Threads.atomic_add!(MULT_NUMBER, 1) + offset - 1) % $(length(MULTIPLIERS)) + 1]),
            one(UInt64) + 2 * ((MULT_NUMBER[] + offset - 1) ÷ $(length(MULTIPLIERS)))
        )
    end
end

function adjust_vector_width(W, T)
    if T == XSH_RR
        W >>= 1
    elseif T == RXS_M_XS
        W = W
    else
        throw("Have not yet added support for $T.")
    end
    W
end





function rand_pcgPCG_RXS_M_XS_int64_quote(N, WV, Nreps)
    output = Expr(:tuple)
    @assert WV ≤ W64
#    WV = min(W, W64)
    # vector_size = 8WV
#    Nreps, r = divrem(W, WV)
#    r == 0 || throw("0 != $W % $WV = $r.")
    q = quote
#        $(Expr(:meta, :inline))
        prng = pointer(rng)
        increment = vbroadcast(Vec{$WV,UInt64}, rng.increment)
    end
    if Nreps > N
        NNrep, rr = divrem(Nreps, N)
        for n ∈ 1:N
            push!(q.args, quote
                $(Symbol(:state_, n)) = vload(Vec{$WV,UInt64}, prng + $(REGISTER_SIZE * (n-1)) )
                $(Symbol(:multiplier_, n)) = vload(Vec{$WV,UInt64}, prng + $(REGISTER_SIZE * (N + n-1)) )
            end)
        end
        i = 0
        for nr ∈ 1:NNrep
            for n ∈ 1:N
                i += 1
                state = Symbol(:state_, n)
                xorshifted = Symbol(:xorshifted_, i)
                count = Symbol(:count_, i)
                out = Symbol(:out_, i)
                push!(q.args, quote
                    $count = vadd(vbroadcast(Vec{$WV,UInt}, UInt(5)), vright_bitshift($state, Val{59}()))
                    $xorshifted = vmul(vxor(
                            vright_bitshift($state, $count), $state
                        ), 0xaef17502108ef2d9)
                    $state = vmuladd($(Symbol(:multiplier_, n)), $state, increment)
                    $out = vxor($xorshifted, vright_bitshift($xorshifted, Val{43}()))
                end)
                push!(output.args, out)
#                for w ∈ 1:WV
#                    push!(output.args, :(@inbounds $out[$w]))
#                end
            end
        end
        for n ∈ 1:rr
            i += 1
            state = Symbol(:state_, n)
            xorshifted = Symbol(:xorshifted_, i)
            count = Symbol(:count_, i)
            out = Symbol(:out_, i)
            push!(q.args, quote
                $count = vadd(vbroadcast(Vec{$WV,UInt}, UInt(5)), vright_bitshift($state, Val{59}()))
                $xorshifted = vmul(vxor(
                        vright_bitshift($state, $count), $state
                    ), 0xaef17502108ef2d9)
                $state = vmuladd($(Symbol(:multiplier_, n)), $state, increment)
                $out = vxor($xorshifted, vright_bitshift($xorshifted, Val{43}()))
            end)
            push!(output.args, out)
#            for w ∈ 1:WV
#                push!(output.args, :(@inbounds $out[$w]))
#            end
        end
        for n ∈ 1:N
            push!(q.args, quote
                vstore!(prng + $(REGISTER_SIZE * (n-1)), $(Symbol(:state_, n)))
            end)
        end
    else # Nreps <= N
        for n ∈ 1:Nreps
            state = Symbol(:state_, n)
            xorshifted = Symbol(:xorshifted_, n)
            count = Symbol(:count_, n)
            out = Symbol(:out_, n)
            push!(q.args, quote
                $state = vload(Vec{$WV,UInt64}, prng + $(REGISTER_SIZE * (n-1)) )
                $count = vadd(vbroadcast(Vec{$WV,UInt}, UInt(5)), vright_bitshift($state, Val{59}()))
                $xorshifted = vmul(vxor(
                        vright_bitshift($state, $count), $state
                    ), 0xaef17502108ef2d9)
                $state = vmuladd(vload(Vec{$WV,UInt64}, prng + $(REGISTER_SIZE * (N + n-1)) ), $state, increment)
                $out = vxor($xorshifted, vright_bitshift($xorshifted, Val{43}()))
                vstore!(prng + $(REGISTER_SIZE * (n-1)), $state)
            end)
            push!(output.args, out)
#            for w ∈ 1:WV
#                push!(output.args, :(@inbounds $out[$w]))
#            end
        end
    end
    push!(q.args, output)
    q
end

@inline rotate(x, r) = x >> r | x << (-r & 31)
#@generated
@inline function rotate(x::Vec{W,T1}, r::Vec{W,T2}) where {W,T1,T2}
#    quote
#        $(Expr(:meta, :inline))
        xshiftright = SIMDPirates.vright_bitshift(x, r)
        nra31 = SIMDPirates.vand(SIMDPirates.vsub(r), SIMDPirates.vbroadcast(Vec{W,T2}, T2(31)))
        xshiftleft = SIMDPirates.vleft_bitshift(x, nra31)
        SIMDPirates.vor(xshiftright, xshiftleft)
#        $(Expr(:tuple, [:(@inbounds Core.VecElement(rotate(x[$w].value, r[$w].value))) for w ∈ 1:W]...))
#    end
end

function rand_pcgPCG_XSH_RR_int32_quote(N, WV, Nreps)
    output = Expr(:tuple)
#    WV = min(W, W64)
    WV32 = 2WV
#    Nreps, r = divrem(W, WV)
    #    r == 0 || throw("0 != $W % $WV = $r.")
    if WV > W64
        throw("WV == $WV > $W64 = W64")
    end
    q = quote
#        $(Expr(:meta, :inline))
        prng = pointer(rng)
        increment = vbroadcast(Vec{$WV,UInt64}, rng.increment)
    end
    if Nreps > N
        NNrep, rr = divrem(Nreps, N)
        for n ∈ 1:N
            push!(q.args, quote
                $(Symbol(:state_, n)) = vload(Vec{$WV,UInt64}, prng + $(REGISTER_SIZE * (n-1)) )
                $(Symbol(:multiplier_, n)) = vload(Vec{$WV,UInt64}, prng + $(REGISTER_SIZE * (N + n-1)) )
            end)
        end
        push!(q.args, :(multiplier = rng.multiplier))
        i = 0
        for nr ∈ 1:NNrep
            for n ∈ 1:N
                i += 1
                state = Symbol(:state_, n)
                xorshifted = Symbol(:xorshifted_, i)
                rot = Symbol(:rot_, i)
                out = Symbol(:out,i)
                push!(q.args, quote
                      $xorshifted = pirate_reinterpret(Vec{$WV32,UInt32}, vright_bitshift(
                        vxor(
                            vright_bitshift($state, vbroadcast(Vec{$WV,UInt}, 18)), $state
                        ), vbroadcast(Vec{$WV,UInt}, 27)
                      ))
                      $rot = pirate_reinterpret(Vec{$WV32,UInt32},vright_bitshift($state, vbroadcast(Vec{$WV,UInt}, 59)))
                      $state = vmuladd($(Symbol(:multiplier_, n)), $state, increment)
#                      $out = rotate($xorshifted, $rot)
                      $out = @inbounds $(Expr(:tuple, [:(Core.VecElement(rotate($xorshifted[$w].value, $rot[$w].value))) for w ∈ 1:2:WV32]... ))
                      end)
                push!(output.args, out)
#               for w ∈ 1:2:WV32
#                    push!(output.args, :(@inbounds Core.VecElement(rotate($xorshifted[$w].value, $rot[$w].value))))
#                end
            end
        end
        for n ∈ 1:rr
            i += 1
            state = Symbol(:state_, n)
            xorshifted = Symbol(:xorshifted_, i)
            rot = Symbol(:rot_, i)
            out = Symbol(:out,i)
            push!(q.args, quote
                $xorshifted = pirate_reinterpret(Vec{$WV32,UInt32}, vright_bitshift(
                    vxor(
                        vright_bitshift($state, vbroadcast(Vec{$WV,UInt}, 18)), $state
                    ), vbroadcast(Vec{$WV,UInt}, 27)
                ))
                $rot = pirate_reinterpret(Vec{$WV32,UInt32},vright_bitshift($state, vbroadcast(Vec{$WV,UInt}, 59)))
                $state = vmuladd($(Symbol(:multiplier_, n)), $state, increment)
                  $out = @inbounds $(Expr(:tuple, [:(Core.VecElement(rotate($xorshifted[$w].value, $rot[$w].value))) for w ∈ 1:2:WV32]... ))
#                $out = rotate($xorshifted, $rot)
            end)
            push!(output.args, out)
#            for w ∈ 1:2:WV32
#                push!(output.args, :(@inbounds Core.VecElement(rotate($xorshifted[$w].value, $rot[$w].value))))
#            end
        end
        for n ∈ 1:N
            push!(q.args, quote
                vstore!(prng + $(REGISTER_SIZE * (n-1)), $(Symbol(:state_, n)))
            end)
        end
    else # Nreps <= N
        for n ∈ 1:Nreps
            state = Symbol(:state_, n)
            xorshifted = Symbol(:xorshifted_, n)
            rot = Symbol(:rot_, n)
            out = Symbol(:out,n)
            push!(q.args, quote
                $state = vload(Vec{$WV,UInt64}, prng + $(REGISTER_SIZE * (n-1)) )
                vstore!(
                    prng + $(REGISTER_SIZE * (n-1)),
                    vmuladd(vload(Vec{$WV,UInt64}, prng + $(REGISTER_SIZE * (N + n-1)) ), $state, increment)
                )
                $xorshifted = pirate_reinterpret(Vec{$WV32,UInt32}, vright_bitshift(
                    vxor(
                        vright_bitshift($state, vbroadcast(Vec{$WV,UInt}, 18)), $state
                    ), vbroadcast(Vec{$WV,UInt}, 27)
                ))
                  $rot = pirate_reinterpret(Vec{$WV32,UInt32},vright_bitshift($state, vbroadcast(Vec{$WV,UInt}, 59)))
#                  $out = rotate($xorshifted, $rot)
                  $out = @inbounds $(Expr(:tuple, [:(Core.VecElement(rotate($xorshifted[$w].value, $rot[$w].value))) for w ∈ 1:2:WV32]... ))

            end)
            push!(output.args, out)
#            for w ∈ 1:2:WV32
#                push!(output.args, :(@inbounds Core.VecElement(rotate($xorshifted[$w].value, $rot[$w].value))))
#            end
        end
    end
    push!(q.args, output)
    q
end







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
function mask_expr(N, U, ::Type{Float64}, x = :x)
    if U == UInt32
        x_expr = :(pirate_reinterpret(NTuple{$N,Core.VecElement{UInt64}}, $x))
    elseif U == UInt64
        x_expr = x
    end
    quote
        pirate_reinterpret(
            NTuple{$N,Core.VecElement{Float64}},
            vor(vand($x_expr, 0x000fffffffffffff), 0x3ff0000000000000)
        )
    end
end
function mask_expr(N, U, ::Type{Float32}, x = :x)
    if U == UInt32
        x_expr = x
    elseif U == UInt64
        x_expr = :(pirate_reinterpret(NTuple{$(2N),Core.VecElement{UInt32}}, $x))
    end
    quote
        pirate_reinterpret(
            NTuple{$(2N),Core.VecElement{Float32}},
            vor(vand(
                $x_expr,
                0x007fffff), 0x3f800000)
        )
    end
end

@generated function mask(x::NTuple{N,Core.VecElement{UInt64}}, ::Type{T}) where {N,T}
    quote
        $(Expr(:meta, :inline))
        $(mask_expr(N, UInt64, T))
    end
end

@generated function Random.rand(rng::PCG{N}, ::Type{Vec{W,UInt32}}) where {N,W}
    quote
        $(Expr(:meta,:inline))
        $(rand_pcgPCG_XSH_RR_int32_quote(N, W, 1))
    end
end
@generated function Random.rand(rng::PCG{N}, ::Type{Vec{W,UInt64}}) where {W,N}
    quote
        $(Expr(:meta,:inline))
        $(rand_pcgPCG_RXS_M_XS_int64_quote(N, W, 1))
    end
end
@generated function Random.rand(rng::PCG{P}, ::Type{NTuple{N,Vec{W,UInt32}}}) where {P,N,W}
    quote
        $(Expr(:meta,:inline))
        $(rand_pcgPCG_XSH_RR_int32_quote(P, W, N))
    end
end
@generated function Random.rand(rng::PCG{P}, ::Type{NTuple{N,Vec{W,UInt64}}}) where {P,W,N}
    quote
        $(Expr(:meta,:inline))
        $(rand_pcgPCG_RXS_M_XS_int64_quote(P, W, N))
    end
end

function rand_pcg_float_quote(N,W,::Type{T},::Type{XSH_RR}) where T
    intsym = gensym(:int)
    masked = gensym(:masked)
    Wadjust = W64 * W ÷ TypeVectorWidth(T)
    quote
        $intsym = $(rand_pcgPCG_XSH_RR_int32_quote(N, Wadjust << 1, 1))
        $masked = $(mask_expr(Wadjust, UInt32, T, intsym))
        vsub($(T(2)), $masked)
    end
end
function rand_pcg_float_quote(N,W,::Type{T},::Type{RXS_M_XS}) where T
    intsym = gensym(:int)
    masked = gensym(:masked)
    Wadjust = W64 * W ÷ TypeVectorWidth(T)
    quote
        $intsym = $(rand_pcgPCG_RXS_M_XS_int64_quote(N, Wadjust, 1))
        $masked = $(mask_expr(Wadjust, UInt64, T, intsym))
        vsub($(T(2)), $masked)
    end
end
@generated function Random.rand(rng::PCG{N}, ::Type{Vec{W,T}}, ::Type{PCG_TYPE}) where {N,W,T,PCG_TYPE<:AbstractPCG_TYPE}
    quote
        $(Expr(:meta, :inline))
        $(rand_pcg_float_quote(N, W, T, PCG_TYPE))
    end
end
@generated function Random.rand(rng::PCG{N}, ::Type{Vec{W,T}}) where {N,W,T}
    quote
        $(Expr(:meta, :inline))
        $(rand_pcg_float_quote(N, W, T, default_pcg_type(W, T)))
    end
end

function rand_pcg_float_quote(P,W,N,::Type{T},::Type{XSH_RR}) where T
    intsym = gensym(:int)
    masked = gensym(:masked)
    Wadjust = W64 * W ÷ TypeVectorWidth(T)
    quote
        $intsym = $(rand_pcgPCG_XSH_RR_int32_quote(P, Wadjust, N << 1))
        vtwo = SIMDPirates.vbroadcast(Vec{$W,$T}, $(T(2)))
        @inbounds $(Expr(:tuple,
               [:(SIMDPirates.vsub(vtwo, $(mask_expr(Wadjust, UInt32, T, :($intsym[$n]))))) for n ∈ 1:(N << 1)]...
        ))
    end
end
function rand_pcg_float_quote(P,W,N,::Type{T},::Type{RXS_M_XS}) where T
    intsym = gensym(:int)
    masked = gensym(:masked)
    Wadjust = W64 * W ÷ TypeVectorWidth(T)
    quote
        $intsym = $(rand_pcgPCG_RXS_M_XS_int64_quote(P, Wadjust, N))
#        $masked = $(mask_expr(Wadjust, UInt64, T, intsym))
#        vsub($(T(2)), $masked)
        vtwo = SIMDPirates.vbroadcast(Vec{$W,$T}, $(T(2)))
        @inbounds $(Expr(:tuple,
               [:(SIMDPirates.vsub(vtwo, $(mask_expr(Wadjust, UInt64, T, :($intsym[$n]))))) for n ∈ 1:N]...
        ))
    end
end
@generated function Random.rand(rng::PCG{P}, ::Type{NTuple{N,Vec{W,T}}}, ::Type{PCG_TYPE}) where {P,N,W,T,PCG_TYPE<:AbstractPCG_TYPE}
    quote
        $(Expr(:meta, :inline))
        $(rand_pcg_float_quote(P, W, N, T, PCG_TYPE))
    end
end
@generated function Random.rand(rng::PCG{P}, ::Type{NTuple{N,Vec{W,T}}}) where {P,N,W,T}
    quote
        $(Expr(:meta, :inline))
        $(rand_pcg_float_quote(P, W, N, T, default_pcg_type(W*N, T)))
    end
end


#function subset_vec(name, W, offset = 0)
#    Expr(:tuple, [:(@inbounds $name[$(offset+w)]) for w ∈ 1:W]...)
#end


function randexp_quote(P, W, N, T, PCG_TYPE)
#    WT = TypeVectorWidth(T)
    #    NW, r = divrem(W, WT)
    WT = W
    output = Expr(:tuple)
    q = quote
#        $(Expr(:meta, :inline))
        u = $(rand_pcg_float_quote(P,W,N,T,PCG_TYPE))
    end
    for n ∈ 1:N
        e_n = Symbol(:e_,n)
        push!(q.args,
            :($e_n = @inbounds vabs(SLEEFPirates.log_fast(u[$n])))
#            :($e_n = vabs(SLEEFPirates.log_fast($(subset_vec(:u,WT,(n-1)*WT)))))
              )
        push!(output.args, e_n)
#        for w ∈ 1:WT
#            push!(output.args, :(@inbounds $e_n[$w]))
#        end
    end
#=    if r > 0
        e_n = Symbol(:e_,NW+1)
        push!(q.args,
            :($e_n = vabs(SLEEFPirates.log_fast(u[$N])))
#            :($e_n = vabs(SLEEFPirates.log_fast($(subset_vec(:u,r,NW*WT)))))
              )
        push!(output.args, $e_n)
#        for w ∈ 1:r
#            push!(output.args, :(@inbounds $e_n[$w]))
#        end
    end=#
    push!(q.args, output)
    q
end

@generated function Random.randexp(rng::PCG{N}, ::Type{Vec{W,T}}, ::Type{PCG_TYPE}) where {W,N,T,PCG_TYPE<:AbstractPCG_TYPE}
    quote
        $(Expr(:meta,:inline))
        out = $(randexp_quote(N, W, 1, T, PCG_TYPE))
        @inbounds out[1]
    end
end
@generated function Random.randexp(rng::PCG{N}, ::Type{Vec{W,T}}) where {N,W,T}
    quote
        $(Expr(:meta,:inline))
        out = $(randexp_quote(N, W, 1, T, default_pcg_type(W, T)))
        @inbounds out[1]
    end
end
@generated function Random.randexp(rng::PCG{P}, ::Type{NTuple{N,Vec{W,T}}}, ::Type{PCG_TYPE}) where {P, N,W,T,PCG_TYPE<:AbstractPCG_TYPE}
    quote
        $(Expr(:meta,:inline))
        $(randexp_quote(P, W, N, T, PCG_TYPE))
    end
end
@generated function Random.randexp(rng::PCG{P}, ::Type{NTuple{N,Vec{W,T}}}) where {N,P,W,T}
    quote
        $(Expr(:meta,:inline))
        $(randexp_quote(P, W, N, T, default_pcg_type(W*N, T)))
    end
end

function randnegexp_quote(P, W, N, T, PCG_TYPE)
    WT = TypeVectorWidth(T)
    NW, r = divrem(W, WT)
    output = Expr(:tuple)
    q = quote
#        $(Expr(:meta, :inline))
        u = $(rand_pcg_float_quote(P,W,N,T,PCG_TYPE))
    end
    for n ∈ 1:NW
        e_n = Symbol(:e_,n)
        push!(q.args,
            :($e_n = SLEEFPirates.log_fast(u[$n]))
#            :($e_n = SLEEFPirates.log_fast($(subset_vec(:u,WT,(n-1)*WT))))
        )
        push!(output.args, e_n)
#        for w ∈ 1:WT
#            push!(output.args, :($e_n[$w]))
#        end
    end
#=    if r > 0
        e_n = Symbol(:e_,NW+1)
        push!(q.args,
            :($e_n = SLEEFPirates.log_fast($(subset_vec(:u,r,NW*WT))))
        )
        for w ∈ 1:r
            push!(output.args, :($e_n[$w]))
        end
    end=#
    push!(q.args, output)
    q
end

@generated function randnegexp(rng::PCG{N}, ::Type{Vec{W,T}}, ::Type{PCG_TYPE}) where {N,W,T,PCG_TYPE<:AbstractPCG_TYPE}
    quote
        $(Expr(:meta,:inline))
        $(randnegexp_quote(N, W, 1, T, PCG_TYPE))
    end
end
@generated function randnegexp(rng::PCG{N}, ::Type{Vec{W,T}}) where {N,W,T}
    quote
        $(Expr(:meta,:inline))
        $(randnegexp_quote(N, W, 1, T, default_pcg_type(W, T)))
    end
end
@generated function randnegexp(rng::PCG{P}, ::Type{NTuple{N,Vec{W,T}}}, ::Type{PCG_TYPE}) where {N,P,W,T,PCG_TYPE<:AbstractPCG_TYPE}
    quote
        $(Expr(:meta,:inline))
        $(randnegexp_quote(P, W, N, T, PCG_TYPE))
    end
end
@generated function randnegexp(rng::PCG{P}, ::Type{NTuple{N,Vec{W,T}}}) where {N,W,T,P}
    quote
        $(Expr(:meta,:inline))
        $(randnegexp_quote(P, W, N, T, default_pcg_type(W*N, T)))
    end
end


function randn_quote(P, W, N, T, PCG_TYPE)
#    @show P,W,N,PCG_TYPE
    #    WT = TypeVectorWidth(T)
    WT = W
#    NW, r = divrem(W >> 1, WT)
    # workaround
    # splitsincos = WT * sizeof(T) < 64

    output = Expr(:tuple)
    q = quote
#        $(Expr(:meta, :inline))
        u = $(rand_pcg_float_quote(P,W,N,T,PCG_TYPE))
        vπ = vbroadcast(Vec{$WT, $T}, $(T(π)))
    end
    for n ∈ 1:N >> 1
        u1_n = Symbol(:u1_, n)
        u2_n = Symbol(:u2_, n)
        # get the vectors u_1 and u_2
        push!(q.args, quote
#            $u1_n = SLEEFPirates.log_fast($(subset_vec(:u,WT,(n-1)*2WT)))
#            $u2_n =                    $(subset_vec(:u,WT,(n-1)*2WT + WT))
            $u1_n = @inbounds SLEEFPirates.log_fast(u[$(2n-1)])
            $u2_n = @inbounds u[$(2n)]
            $u1_n = vsqrt( vabs( vadd($u1_n, $u1_n) ) )
            $u2_n = vadd($u2_n, $u2_n)
        end)
        s_n = Symbol(:s_, n)
        c_n = Symbol(:c_, n)
        # workaround for https://github.com/JuliaLang/julia/issues/30426
        # if splitsincos
        push!(q.args, :(($s_n, $c_n) = SLEEFPirates.sincos_fast(vmul($u2_n, vπ))) )
        # else
        #     sc_n = Symbol(:sc_, n)
        #     push!(q.args,  quote
        #         $sc_n = SLEEFPirates.sincos_fast(vmul($u2_n, vπ))
        #         $s_n = $(subset_vec(sc_n, WT, 0))
        #         $c_n = $(subset_vec(sc_n, WT, WT))
        #     end)
        # end
        z1_n = Symbol(:z1_,n)
        z2_n = Symbol(:z2_,n)
        push!(q.args, quote
            $z1_n = vmul($u1_n, $s_n)
            $z2_n = vmul($u1_n, $c_n)
        end )
        push!(output.args, z1_n)
        push!(output.args, z2_n)
#=        for w ∈ 1:WT
            push!(output.args, :(@inbounds $z1_n[$w]))
        end
        for w ∈ 1:WT
            push!(output.args, :(@inbounds $z2_n[$w]))
        end=#
    end
    if isodd(N)
        n = (N >> 1) + 1
        u1_n = Symbol(:u1_, n)
        u2_n = Symbol(:u2_, n)
        # get the vectors u_1 and u_2
        push!(q.args, quote
              u_odd = @inbounds u[$N]
            $u1_n = @inbounds SLEEFPirates.log_fast($(Expr(:tuple, [:(u_odd[$w]) for w ∈ 1:(W>>1)]...)))
            $u2_n = @inbounds $(Expr(:tuple, [:(u_odd[$w]) for w ∈ ((W>>1)+1):W]...))
#            $u1_n = SLEEFPirates.log_fast($(subset_vec(:u,r,NW*2WT)))
#            $u2_n =                    $(subset_vec(:u,r,NW*2WT+r))
            $u1_n = vsqrt( vabs( vadd($u1_n, $u1_n) ) )
            $u2_n = vadd($u2_n, $u2_n)
        end)
        s_n = Symbol(:s_, n)
        c_n = Symbol(:c_, n)
        # workaround for https://github.com/JuliaLang/julia/issues/30426
        # AFAIK r * sizeof(T) < 64 for all supported use cases
        # if r * sizeof(T) < 64
        push!(q.args, :(($s_n, $c_n) = SLEEFPirates.sincos_fast(vmul($u2_n, $(T(π)))) ))
        # else
        #     sc_n = Symbol(:sc_, NW+1)
        #     push!(q.args,  quote
        #         $sc_n = SLEEFPirates.sincos_fast(vmul($u2_n, vπ))
        #         $s_n = $(subset_vec(sc_n, r, 0))
        #         $c_n = $(subset_vec(sc_n, r, r))
        #     end)
        # end
#        z1_n = Symbol(:z1_,n)
#        z2_n = Symbol(:z2_,n)
        u_n = Symbol(:u_,n)
        sc_n = Symbol(:sc_,n)
        z_n = Symbol(:z_,n)

        push!(q.args, quote
              $u_n = $(Expr(:tuple, [:($u1_n[$w]) for w ∈ 1:(W>>1)]..., [:($u1_n[$w]) for w ∈ 1:(W>>1)]...))
              $sc_n = $(Expr(:tuple, [:($s_n[$w]) for w ∈ 1:(W>>1)]..., [:($c_n[$w]) for w ∈ 1:(W>>1)]...))
              $z_n = vmul($u_n, $sc_n)
#            $z1_n = vmul($u1_n, $s_n)
#            $z2_n = vmul($u1_n, $c_n)
        end )
#        push!(output.args, Expr(:tuple, [:($z1_n[$w]) for w ∈ 1:(W>>1)]..., [:($z2_n[$w]) for w ∈ 1:(W>>1)]...))
        push!(output.args, z_n)
#=        for w ∈ 1:r
            push!(output.args, :(@inbounds $z1_n[$w]))
        end
        for w ∈ 1:r
            push!(output.args, :(@inbounds $z2_n[$w]))
        end=#
    end
    push!(q.args, output)
    q
end

@generated function Random.randn(rng::PCG{N}, ::Type{Vec{W,T}}, ::Type{PCG_TYPE}) where {N,W,T,PCG_TYPE<:AbstractPCG_TYPE}
    quote
        $(Expr(:meta,:inline))
        z = $(randn_quote(N, W, 1, T, PCG_TYPE))
        @inbounds z[1]
    end
end
@generated function Random.randn(rng::PCG{N}, ::Type{Vec{W,T}}) where {W,N,T}
    quote
        $(Expr(:meta,:inline))
        z = $(randn_quote(N, W, 1, T, default_pcg_type(W, T)))
        @inbounds z[1]
    end
end
@generated function Random.randn(rng::PCG{P}, ::Type{NTuple{N,Vec{W,T}}}, ::Type{PCG_TYPE}) where {N,P,W,T,PCG_TYPE<:AbstractPCG_TYPE}
    quote
        $(Expr(:meta,:inline))
        $(randn_quote(P, W, N, T, PCG_TYPE))
    end
end
@generated function Random.randn(rng::PCG{P}, ::Type{NTuple{N,Vec{W,T}}}) where {N,P,W,T}
    quote
        $(Expr(:meta,:inline))
        $(randn_quote(P, W, N, T, default_pcg_type(W*N, T)))
    end
end

function rand_loop_quote(N, T, rngexpr)
    W = TypeVectorWidth(T)
    quote
        ptr_A = pointer(A)
        L = length(x)
        nrep, nrem = divrem(L, $(N*W))
        GC.@preserve A begin
            for i ∈ 0:nrep-1
                r = $rngexpr(rng, NTuple{$N,Vec{$W,$T}})
                Base.Cartesian.@nexprs $N n -> begin
                    @inbounds SIMDPirates.vstore!(ptr_A + $(sizeof(T)*W) * ( (n-1) + $(N)*i ), r[n])
                end
            end
            nrem == 0 && return A
            r = $rngexpr(rng, NTuple{$N,Vec{$W,$T}})
            nremrep, nremrem = divrem(nrem, $W)
            if nremrep > 0
                Base.Cartesian.@nexprs $nremrep n -> begin
                    @inbounds SIMDPirates.vstore!(ptr_A + $(sizeof(T)*W) * ( (n-1) + $(N)*nrep ), r[n])
                end
            end
            if nremrem > 0
                @inbounds SIMDPirates.vstore!(ptr_A + $(sizeof(T)*W) * ( (n-1) + $(N)*nrep + nremrep ), r[1+nremrep], $(mask_type(T))(2^nremrem-1))
            end
            return A
        end
    end
    
end

@generated function Random.rand!(rng::PCG{N}, x::AbstractArray{T}, ::Type{PCG_TYPE} = RXS_M_XS) where {N,T <: Real, PCG_TYPE <: AbstractPCG_TYPE}
    rand_loop_quote(adjust_vector_width(N, PCG_TYPE), T, :rand)
end
@generated function Random.randexp!(rng::PCG{N}, x::AbstractArray{T}, ::Type{PCG_TYPE} = RXS_M_XS) where {N,T <: Real, PCG_TYPE <: AbstractPCG_TYPE}
    rand_loop_quote(adjust_vector_width(N, PCG_TYPE), T, :randexp)
end
@generated function Random.randn!(rng::PCG{N}, x::AbstractArray{T}, ::Type{PCG_TYPE} = RXS_M_XS) where {N,T <: Real, PCG_TYPE <: AbstractPCG_TYPE}
    rand_loop_quote(adjust_vector_width(N, PCG_TYPE), T, :randn)
end

#=
function unrolled_rand_quote(NWT, rand_expr, store_expr)
    quote
        L = length(x)
        ptr_x = pointer(x)
        L_o_NWT, r = divrem(L, $NWT)
        for i ∈ 1:$NWT:(L_o_NWT*$NWT)
            u = $rand_expr
            $store_expr
        end
        if r > 0
            u = $rand_expr
            j = 1
            @inbounds for i in L-r+1:L
                x[i] = u[j].value
                j += 1
            end
        end
        x
    end
end


@generated function Random.rand!(rng::PCG{N}, x::AbstractArray{T}, ::Type{PCG_TYPE} = RXS_M_XS) where {N,T <: Real, PCG_TYPE <: AbstractPCG_TYPE}
    WT = TypeVectorWidth(T)
    Nhalf = adjust_vector_width(N, PCG_TYPE)
    NWT = Nhalf*WT
    # float_q = rand_pcg_float_quote(N,NWT,T)
    float_q = :(rand(rng, Vec{$NWT,$T}, $PCG_TYPE))
    store_expr = quote end
    for n ∈ 0:Nhalf-1
        push!(store_expr.args, :(vstore!(ptr_x, i + $(n*WT), $(subset_vec(:u, WT, n*WT)))))
    end
    unrolled_rand_quote(NWT, float_q, store_expr)
end
@generated function Random.randexp!(rng::PCG{N}, x::AbstractArray{T}, ::Type{PCG_TYPE} = RXS_M_XS) where {N,T <: Real, PCG_TYPE <: AbstractPCG_TYPE}
    WT = TypeVectorWidth(T)
    Nhalf = adjust_vector_width(N, PCG_TYPE)
    NWT = Nhalf*WT
    store_expr = quote end
    for n ∈ 0:Nhalf-1
        push!(store_expr.args, :(vstore!(ptr_x, $(subset_vec(:u, WT, n*WT)), i + $(n*WT))))
    end
    unrolled_rand_quote(NWT, :(randexp(rng, Vec{$NWT,$T})), store_expr)
end
@generated function Random.randn!(rng::PCG{N}, x::AbstractArray{T}, ::Type{PCG_TYPE} = RXS_M_XS) where {N,T <: Real, PCG_TYPE <: AbstractPCG_TYPE}
    WT = TypeVectorWidth(T)
    Nhalf = adjust_vector_width(N, PCG_TYPE)
    NWT = Nhalf*WT
    store_expr = quote end
    for n ∈ 0:Nhalf-1
        push!(store_expr.args, :(vstore!(ptr_x, $(subset_vec(:u, WT, n*WT)), i + $(n*WT))))
    end
    unrolled_rand_quote(NWT, :(randn(rng, Vec{$NWT,$T})), store_expr)
end
=#
