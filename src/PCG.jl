

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

abstract type  AbstractPCG{N} end

mutable struct PCG{N} <: AbstractPCG{N}
    state::NTuple{N,Vec{W64,UInt64}}
    multiplier::NTuple{N,Vec{W64,UInt64}}
    increment::Vec{W64,UInt64}
end

@inline Base.pointer(rng::AbstractPCG) = Base.unsafe_convert(Ptr{UInt64}, pointer_from_objref(rng))
# @inline Base.pointer(rng::AbstractPCG) = Base.unsafe_convert(Ptr{Vec{W64,UInt64}}, pointer_from_objref(rng))



@generated function PCG{N}() where N
    quote
        PCG(
            (Base.Cartesian.@ntuple $N n -> (Base.Cartesian.@ntuple $W64 w -> Core.VecElement(rand(UInt64)))),
            (Base.Cartesian.@ntuple $N n -> multipliers[Base.Threads.atomic_add!(MULT_NUMBER, 1)]),
            (Base.Cartesian.@ntuple $W64 w -> Core.VecElement(one(UInt64)))
        )
    end
end


function adjust_vector_width(W, @nospecialize T)
    if T == XSH_RR
        W >>= 1
    elseif T == RXS_M_XS
        W = W
    else
        throw("Have not yet added support for $T.")
    end
    W
end

# adjust_vector_width(W, ::Type{<:AbstractPCG_XSH_RR}) = W >> 1
# adjust_vector_width(W, ::Type{<:AbstractPCG_RXS_M_XS}) = W



@inline rotate(x, r) = x >> r | x << (-r & 31)
@generated function rotate(x::Vec{W,T1}, r::Vec{W,T2}) where {W,T1,T2}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:tuple, [:(@inbounds Core.VecElement(rotate(x[$w].value, r[$w].value))) for w ∈ 1:W]...))
    end
end


function rand_pcgPCG_RXS_M_XS_int64_quote(N, W)
    output = Expr(:tuple)
    Nreps, r = divrem(W, W64)
    r == 0 || throw("0 != $W % $W64 = $r.")
    q = quote
        $(Expr(:meta, :inline))
        prng = pointer(rng)
        increment = rng.increment
    end
    if Nreps > N
        NNrep, rr = divrem(Nreps, N)
        for n ∈ 1:N
            push!(q.args, quote
                $(Symbol(:state_, n)) = vload(Vec{W64,UInt64}, prng + $(REGISTER_SIZE * (n-1)) )
                $(Symbol(:multiplier_, n)) = vload(Vec{W64,UInt64}, prng + $(REGISTER_SIZE * (N + n-1)) )
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
                    $count = vadd(UInt(5), vright_bitshift($state, UInt(59)))
                    $xorshifted = vmul(vxor(
                            vright_bitshift($state, $count), $state
                        ), 0xaef17502108ef2d9)
                    $state = vmuladd($(Symbol(:multiplier_, n)), $state, increment)
                    $out = vxor($xorshifted, vright_bitshift($xorshifted, UInt(43)))
                end)
                for w ∈ 1:W64
                    push!(output.args, :(@inbounds $out[$w]))
                end
            end
        end
        for n ∈ 1:rr
            i += 1
            state = Symbol(:state_, n)
            xorshifted = Symbol(:xorshifted_, i)
            count = Symbol(:count_, i)
            out = Symbol(:out_, i)
            push!(q.args, quote
                $count = vadd(UInt(5), vright_bitshift($state, UInt(59)))
                $xorshifted = vmul(vxor(
                        vright_bitshift($state, $count), $state
                    ), 0xaef17502108ef2d9)
                $state = vmuladd($(Symbol(:multiplier_, n)), $state, increment)
                $out = vxor($xorshifted, vright_bitshift($xorshifted, UInt(43)))
            end)
            for w ∈ 1:W64
                push!(output.args, :(@inbounds $out[$w]))
            end
        end
        for n ∈ 1:N
            push!(q.args, quote
                vstore($(Symbol(:state_, n)),  prng + $(REGISTER_SIZE * (n-1)))
            end)
        end
    else # Nreps <= N
        for n ∈ 1:Nreps
            state = Symbol(:state_, n)
            xorshifted = Symbol(:xorshifted_, n)
            count = Symbol(:count_, n)
            out = Symbol(:out_, n)
            push!(q.args, quote
                $state = vload(Vec{W64,UInt64}, prng + $(REGISTER_SIZE * (n-1)) )
                $count = vadd(UInt(5), vright_bitshift($state, UInt(59)))
                $xorshifted = vmul(vxor(
                        vright_bitshift($state, $count), $state
                    ), 0xaef17502108ef2d9)
                $state = vmuladd(vload(Vec{W64,UInt64}, prng + $(REGISTER_SIZE * (N + n-1)) ), $state, increment)
                $out = vxor($xorshifted, vright_bitshift($xorshifted, UInt(43)))
                vstore($state,  prng + $(REGISTER_SIZE * (n-1)))
            end)
            for w ∈ 1:W64
                push!(output.args, :(@inbounds $out[$w]))
            end
        end
    end
    push!(q.args, output)
    q
end

function rand_pcgPCG_XSH_RR_int32_quote(N, W)
    output = Expr(:tuple)
    Nreps, r = divrem(W, W64)
    r == 0 || throw("0 != $W % $W64 = $r.")
    q = quote
        $(Expr(:meta, :inline))
        prng = pointer(rng)
        increment = rng.increment
    end
    if Nreps > N
        NNrep, rr = divrem(Nreps, N)
        for n ∈ 1:N
            push!(q.args, quote
                $(Symbol(:state_, n)) = vload(Vec{W64,UInt64}, prng + $(REGISTER_SIZE * (n-1)) )
                $(Symbol(:multiplier_, n)) = vload(Vec{W64,UInt64}, prng + $(REGISTER_SIZE * (N + n-1)) )
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
                push!(q.args, quote
                    $xorshifted = pirate_reinterpret(Vec{$W32,UInt32}, vright_bitshift(
                        vxor(
                            vright_bitshift($state, 18), $state
                        ), 27
                    ))
                    $rot = pirate_reinterpret(Vec{$W32,UInt32},vright_bitshift($state, 59))
                    $state = vmuladd($(Symbol(:multiplier_, n)), $state, increment)
                end)
                for w ∈ 1:2:W32
                    push!(output.args, :(@inbounds Core.VecElement(rotate($xorshifted[$w].value, $rot[$w].value))))
                end
            end
        end
        for n ∈ 1:rr
            i += 1
            state = Symbol(:state_, n)
            xorshifted = Symbol(:xorshifted_, i)
            rot = Symbol(:rot_, i)
            push!(q.args, quote
                $xorshifted = pirate_reinterpret(Vec{$W32,UInt32}, vright_bitshift(
                    vxor(
                        vright_bitshift($state, 18), $state
                    ), 27
                ))
                $rot = pirate_reinterpret(Vec{$W32,UInt32},vright_bitshift($state, 59))
                $state = vmuladd($(Symbol(:multiplier_, n)), $state, increment)
            end)
            for w ∈ 1:2:W32
                push!(output.args, :(@inbounds Core.VecElement(rotate($xorshifted[$w].value, $rot[$w].value))))
            end
        end
        for n ∈ 1:N
            push!(q.args, quote
                vstore($(Symbol(:state_, n)),  prng + $(REGISTER_SIZE * (n-1)))
            end)
        end
    else # Nreps <= N
        for n ∈ 1:Nreps
            state = Symbol(:state_, n)
            xorshifted = Symbol(:xorshifted_, n)
            rot = Symbol(:rot_, n)
            push!(q.args, quote
                $state = vload(Vec{W64,UInt64}, prng + $(REGISTER_SIZE * (n-1)) )
                vstore(
                    vmuladd(vload(Vec{W64,UInt64}, prng + $(REGISTER_SIZE * (N + n-1)) ), $state, increment),
                    prng + $(REGISTER_SIZE * (n-1))
                )
                $xorshifted = pirate_reinterpret(Vec{$W32,UInt32}, vright_bitshift(
                    vxor(
                        vright_bitshift($state, 18), $state
                    ), 27
                ))
                $rot = pirate_reinterpret(Vec{$W32,UInt32},vright_bitshift($state, 59))
            end)
            for w ∈ 1:2:W32
                push!(output.args, :(@inbounds Core.VecElement(rotate($xorshifted[$w].value, $rot[$w].value))))
            end
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
    rand_pcgPCG_XSH_RR_int32_quote(N, W)
end
@generated function Random.rand(rng::PCG{N}, ::Type{Vec{W,UInt64}}) where {W,N}
    rand_pcgPCG_RXS_M_XS_int64_quote(N, W)
end

function rand_pcg_float_quote(N,W,::Type{T},::Type{XSH_RR}) where T
    intsym = gensym(:int)
    masked = gensym(:masked)
    Wadjust = W64 * W ÷ TypeVectorWidth(T)
    quote
        $intsym = $(rand_pcgPCG_XSH_RR_int32_quote(N, Wadjust << 1))
        $masked = $(mask_expr(Wadjust, UInt32, T, intsym))
        vsub($(T(2)), $masked)
    end
end
function rand_pcg_float_quote(N,W,::Type{T},::Type{RXS_M_XS}) where T
    intsym = gensym(:int)
    masked = gensym(:masked)
    Wadjust = W64 * W ÷ TypeVectorWidth(T)
    quote
        $intsym = $(rand_pcgPCG_RXS_M_XS_int64_quote(N, Wadjust))
        $masked = $(mask_expr(Wadjust, UInt64, T, intsym))
        vsub($(T(2)), $masked)
    end
end
@generated function Random.rand(rng::PCG{N}, ::Type{Vec{W,T}}, ::Type{PCG_TYPE}) where {N,W,T,PCG_TYPE<:AbstractPCG_TYPE}
    quote
        # $(Expr(:meta, :inline))
        $(rand_pcg_float_quote(N, W, T, PCG_TYPE))
    end
end
@generated function Random.rand(rng::PCG{N}, ::Type{Vec{W,T}}) where {N,W,T}
    quote
        # $(Expr(:meta, :inline))
        $(rand_pcg_float_quote(N, W, T, default_pcg_type(W, T)))
    end
end


function subset_vec(name, W, offset = 0)
    Expr(:tuple, [:(@inbounds $name[$(offset+w)]) for w ∈ 1:W]...)
end


function randexp_quote(N, W, T, PCG_TYPE)
    WT = TypeVectorWidth(T)
    NW, r = divrem(W, WT)
    output = Expr(:tuple)
    q = quote
        $(Expr(:meta, :inline))
        u = $(rand_pcg_float_quote(N,W,T,PCG_TYPE))
    end
    for n ∈ 1:NW
        e_n = Symbol(:e_,n)
        push!(q.args,
            :($e_n = vabs(SLEEF.log_fast($(subset_vec(:u,WT,(n-1)*WT)))))
        )
        for w ∈ 1:WT
            push!(output.args, :(@inbounds $e_n[$w]))
        end
    end
    if r > 0
        e_n = Symbol(:e_,NW+1)
        push!(q.args,
            :($e_n = vabs(SLEEF.log_fast($(subset_vec(:u,r,NW*WT)))))
        )
        for w ∈ 1:r
            push!(output.args, :(@inbounds $e_n[$w]))
        end
    end
    push!(q.args, output)
    q
end

@generated function Random.randexp(rng::PCG{N}, ::Type{Vec{W,T}}, ::Type{PCG_TYPE}) where {N,W,T,PCG_TYPE<:AbstractPCG_TYPE}
    randexp_quote(N, W, T, PCG_TYPE)
end
@generated function Random.randexp(rng::PCG{N}, ::Type{Vec{W,T}}) where {N,W,T}
    randexp_quote(N, W, T, default_pcg_type(W, T))
end

function randnegexp_quote(N, W, T, PCG_TYPE)
    WT = TypeVectorWidth(T)
    NW, r = divrem(W, WT)
    output = Expr(:tuple)
    q = quote
        $(Expr(:meta, :inline))
        u = $(rand_pcg_float_quote(N,W,T,PCG_TYPE))
    end
    for n ∈ 1:NW
        e_n = Symbol(:e_,n)
        push!(q.args,
            :($e_n = SLEEF.log_fast($(subset_vec(:u,WT,(n-1)*WT))))
        )
        for w ∈ 1:WT
            push!(output.args, :($e_n[$w]))
        end
    end
    if r > 0
        e_n = Symbol(:e_,NW+1)
        push!(q.args,
            :($e_n = SLEEF.log_fast($(subset_vec(:u,r,NW*WT))))
        )
        for w ∈ 1:r
            push!(output.args, :($e_n[$w]))
        end
    end
    push!(q.args, output)
    q
end

@generated function randnegexp(rng::PCG{N}, ::Type{Vec{W,T}}, ::Type{PCG_TYPE}) where {N,W,T,PCG_TYPE<:AbstractPCG_TYPE}
    randnegexp_quote(N, W, T, PCG_TYPE)
end
@generated function randnegexp(rng::PCG{N}, ::Type{Vec{W,T}}) where {N,W,T}
    randnegexp_quote(N, W, T, default_pcg_type(W, T))
end

function randn_quote(N, W, T, PCG_TYPE)
    WT = TypeVectorWidth(T)
    NW, r = divrem(W >> 1, WT)
    # workaround
    # splitsincos = WT * sizeof(T) < 64

    output = Expr(:tuple)
    q = quote
        $(Expr(:meta, :inline))
        u = $(rand_pcg_float_quote(N,W,T,PCG_TYPE))
        vπ = vbroadcast(Vec{$WT, $T}, $(T(π)))
    end
    for n ∈ 1:NW
        u1_n = Symbol(:u1_, n)
        u2_n = Symbol(:u2_, n)
        # get the vectors u_1 and u_2
        push!(q.args, quote
            $u1_n = SLEEF.log_fast($(subset_vec(:u,WT,(n-1)*2WT)))
            $u2_n =                    $(subset_vec(:u,WT,(n-1)*2WT + WT))
            $u1_n = vsqrt( vabs( vadd($u1_n, $u1_n) ) )
            $u2_n = vadd($u2_n, $u2_n)
        end)
        s_n = Symbol(:s_, n)
        c_n = Symbol(:c_, n)
        # workaround for https://github.com/JuliaLang/julia/issues/30426
        # if splitsincos
        push!(q.args, :(($s_n, $c_n) = SLEEF.sincos_fast(vmul($u2_n, vπ))) )
        # else
        #     sc_n = Symbol(:sc_, n)
        #     push!(q.args,  quote
        #         $sc_n = SLEEF.sincos_fast(vmul($u2_n, vπ))
        #         $s_n = $(subset_vec(sc_n, WT, 0))
        #         $c_n = $(subset_vec(sc_n, WT, WT))
        #     end)
        # end
        z1_n = Symbol(:z1_,n)
        z2_n = Symbol(:z2_,n)
        push!(q.args, quote
            $z1_n = extract_data(vmul($u1_n, $s_n))
            $z2_n = extract_data(vmul($u1_n, $c_n))
        end )
        for w ∈ 1:WT
            push!(output.args, :(@inbounds $z1_n[$w]))
        end
        for w ∈ 1:WT
            push!(output.args, :(@inbounds $z2_n[$w]))
        end
    end
    if r > 0
        u1_n = Symbol(:u1_, NW+1)
        u2_n = Symbol(:u2_, NW+1)
        # get the vectors u_1 and u_2
        push!(q.args, quote
            $u1_n = SLEEF.log_fast($(subset_vec(:u,r,NW*2WT)))
            $u2_n =                    $(subset_vec(:u,r,NW*2WT+r))
            $u1_n = vsqrt( vabs( vadd($u1_n, $u1_n) ) )
            $u2_n = vadd($u2_n, $u2_n)
        end)
        s_n = Symbol(:s_, NW+1)
        c_n = Symbol(:c_, NW+1)
        # workaround for https://github.com/JuliaLang/julia/issues/30426
        # AFAIK r * sizeof(T) < 64 for all supported use cases
        # if r * sizeof(T) < 64
        push!(q.args, :(($s_n, $c_n) = SLEEF.sincos_fast(vmul($u2_n, vπ)) ))
        # else
        #     sc_n = Symbol(:sc_, NW+1)
        #     push!(q.args,  quote
        #         $sc_n = SLEEF.sincos_fast(vmul($u2_n, vπ))
        #         $s_n = $(subset_vec(sc_n, r, 0))
        #         $c_n = $(subset_vec(sc_n, r, r))
        #     end)
        # end
        z1_n = Symbol(:z1_,NW+1)
        z2_n = Symbol(:z2_,NW+1)
        push!(q.args, quote
            $z1_n = extract_data(vmul($u1_n, $s_n))
            $z2_n = extract_data(vmul($u1_n, $c_n))
        end )
        for w ∈ 1:r
            push!(output.args, :(@inbounds $z1_n[$w]))
        end
        for w ∈ 1:r
            push!(output.args, :(@inbounds $z2_n[$w]))
        end
    end
    push!(q.args, output)
    q
end

@generated function Random.randn(rng::PCG{N}, ::Type{Vec{W,T}}, ::Type{PCG_TYPE}) where {N,W,T,PCG_TYPE<:AbstractPCG_TYPE}
    randn_quote(N, W, T, PCG_TYPE)
end

@generated function Random.randn(rng::PCG{N}, ::Type{Vec{W,T}}) where {W,N,T}
    randn_quote(N, W, T, default_pcg_type(W, T))
end

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
        push!(store_expr.args, :(vstore($(subset_vec(:u, WT, n*WT)), ptr_x, i + $(n*WT))))
    end
    unrolled_rand_quote(NWT, float_q, store_expr)
end
@generated function Random.randexp!(rng::PCG{N}, x::AbstractArray{T}, ::Type{PCG_TYPE} = RXS_M_XS) where {N,T <: Real, PCG_TYPE <: AbstractPCG_TYPE}
    WT = TypeVectorWidth(T)
    Nhalf = adjust_vector_width(N, PCG_TYPE)
    NWT = Nhalf*WT
    store_expr = quote end
    for n ∈ 0:Nhalf-1
        push!(store_expr.args, :(vstore($(subset_vec(:u, WT, n*WT)), ptr_x, i + $(n*WT))))
    end
    unrolled_rand_quote(NWT, :(randexp(rng, Vec{$NWT,$T})), store_expr)
end
@generated function Random.randn!(rng::PCG{N}, x::AbstractArray{T}, ::Type{PCG_TYPE} = RXS_M_XS) where {N,T <: Real, PCG_TYPE <: AbstractPCG_TYPE}
    WT = TypeVectorWidth(T)
    Nhalf = adjust_vector_width(N, PCG_TYPE)
    NWT = Nhalf*WT
    store_expr = quote end
    for n ∈ 0:Nhalf-1
        push!(store_expr.args, :(vstore($(subset_vec(:u, WT, n*WT)), ptr_x, i + $(n*WT))))
    end
    unrolled_rand_quote(NWT, :(randn(rng, Vec{$NWT,$T})), store_expr)
end
