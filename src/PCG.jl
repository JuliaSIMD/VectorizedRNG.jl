
const W64 = REGISTER_SIZE >> 3
const W32 = REGISTER_SIZE >> 2
const W16 = REGISTER_SIZE >> 1

TypeVectorWidth(::Type{Float64}) = W64
TypeVectorWidth(::Type{Float32}) = W32
TypeVectorWidth(::Type{UInt64}) = W64
TypeVectorWidth(::Type{Int64}) = W64
TypeVectorWidth(::Type{UInt32}) = W32
TypeVectorWidth(::Type{Int32}) = W32
TypeVectorWidth(::Type{T}) where T = REGISTER_SIZE ÷ sizeof(T)

abstract type AbstractPCG{N} <: Random.AbstractRNG end
abstract type AbstractPCG_XSH_RR{N} <: AbstractPCG{N} end

mutable struct PCGCore{N} <: AbstractPCG_XSH_RR{N}
    state::NTuple{N,Vec{W64,UInt64}}
    increment::NTuple{N,Vec{W64,UInt64}}
    multiplier::Vec{W64,UInt64}
end
mutable struct PCGcached64{N,W64_} <: AbstractPCG_XSH_RR{N}
    state::NTuple{N,Vec{W64,UInt64}}
    increment::NTuple{N,Vec{W64,UInt64}}
    multiplier::Vec{W64,UInt64}
    uniform64::Vec{W64_,Float64}
    exponential64::Vec{W64_,Float64}
    normal64::Vec{W64_,Float64}
    uniformcount64::Int
    exponentialcount64::Int
    normalcount64::Int
end
mutable struct PCGcached32{N,W32_} <: AbstractPCG_XSH_RR{N}
    state::NTuple{N,Vec{W64,UInt64}}
    increment::NTuple{N,Vec{W64,UInt64}}
    multiplier::Vec{W64,UInt64}
    uniform32::Vec{W32_,Float32}
    exponential32::Vec{W32_,Float32}
    normal32::Vec{W32_,Float32}
    uniformcount32::Int
    exponentialcount32::Int
    normalcount32::Int
end
mutable struct PCGcached{N,W64_,W32_} <: AbstractPCG_XSH_RR{N}
    state::NTuple{N,Vec{W64,UInt64}}
    increment::NTuple{N,Vec{W64,UInt64}}
    multiplier::Vec{W64,UInt64}
    uniform64::Vec{W64_,Float64}
    exponential64::Vec{W64_,Float64}
    normal64::Vec{W64_,Float64}
    uniform32::Vec{W32_,Float32}
    exponential32::Vec{W32_,Float32}
    normal32::Vec{W32_,Float32}
    uniformcount64::Int
    exponentialcount64::Int
    normalcount64::Int
    uniformcount32::Int
    exponentialcount32::Int
    normalcount32::Int
    function PCGcached{N,W64_,W32_}(state::NTuple{N,Vec{W64,UInt64}}, increment::NTuple{N,Vec{W64,UInt64}}, multiplier::Vec{W64,UInt64}) where {N,W64_,W32_}
        rng = new{N,W64_,W32_}(state, increment, multiplier)
        rng.uniformcount64 = W64_
        rng.exponentialcount64 = W64_
        rng.normalcount64 = W64_
        rng.uniformcount32 = W32_
        rng.exponentialcount32 = W32_
        rng.normalcount32 = W32_
        rng
    end
end
const PCG_cached_64{N,W64_} = Union{PCGcached64{N,W64_}, PCGcached{N,W64_}}
const PCG_cached_32{N,W32_} = Union{PCGcached32{N,W32_}, PCGcached{N,<:Any,W32_}}

@inline Base.pointer(rng::AbstractPCG) = Base.unsafe_convert(Ptr{Vec{W64,UInt64}}, pointer_from_objref(rng))

cache64_offset(::Type{PCGcached{N,W64_,W32_}}) where {N,W64_,W32_} = cache64_offset(N)
cache32_offset(::Type{PCGcached{N,W64_,W32_}}) where {N,W64_,W32_} = cache32_offset(N,W64_)
cache32_offset(::Type{PCGcached32{N,W32_}}) where {N,W32_} = cache64_offset(N)
@noinline function cache64_offset(N)
    W64 * (2N+1) * sizeof(UInt64)
end
@noinline function cache32_offset(N, W64_)
    (W64 * (2N+1) + 3W64_) * sizeof(UInt64)
end

function PCG4Core(state::NTuple{N,Vec{W64,UInt64}}, increment::NTuple{N,Vec{W64,UInt64}}) where N
    PCGCore(
        state, increment, vbroadcast(Vec{W64,UInt64}, 6364136223846793005)
    )
end



"""
Rounds integers up to the nearest odd integer.
"""
make_odd(x::Integer) = x | 0x01
function PCGCore(state::NTuple{N,Vec{W64,UInt64}}) where N
    step = typemax(Int) ÷ (N*W64)
    PCGCore(
        state,
        ntuple(i -> ntuple(inc -> Core.VecElement(reinterpret(UInt, make_odd((W64*(i-1)+inc)*step ))), Val(W64)), Val(N)),
        vbroadcast(Vec{W64,UInt64}, 0x5851f42d4c957f2d)
    )
end

function PCGCore{N}(rng = Random.GLOBAL_RNG) where N # create random initial state
    PCGCore(ntuple(i -> ntuple(s -> Core.VecElement(rand(rng, UInt64)), Val(W64)), Val(N)))
end
@generated function PCGcached(state::NTuple{N,Vec{W64,UInt64}}) where N
    W32_ = N * W64
    W64_ = W32_ >> 1
    :(PCGcached{$N,$W64_,$W32_}(state))
end
function PCGcached{N,W64_,W32_}(state::NTuple{N,Vec{W64,UInt64}}) where {N,W32_,W64_}
    step = typemax(Int) ÷ (N*W64)
    PCGcached{N,W64_,W32_}(
        state,
        ntuple(i -> ntuple(inc -> Core.VecElement(reinterpret(UInt, make_odd((W64*(i-1)+inc)*step ))), Val(W64)), Val(N)),
        vbroadcast(Vec{W64,UInt64}, 6364136223846793005)
    )
end
function PCGcached{N}(rng = Random.GLOBAL_RNG) where N # create random initial state
    PCGcached(ntuple(i -> ntuple(s -> Core.VecElement(rand(rng, UInt64)), Val(W64)), Val(N)))
end
function PCGcached{N,W64_,W32_}(rng = Random.GLOBAL_RNG) where {N,W32_,W64_} # create random initial state
    PCGcached{N,W64_,W32_}(ntuple(i -> ntuple(s -> Core.VecElement(rand(rng, UInt64)), Val(W64)), Val(N)))
end

@inline rotate(x, r) = x >> r | x << (-r & 31)
@generated function rotate(x::Vec{W,T1}, r::Vec{W,T2}) where {W,T1,T2}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:tuple, [:(VE(rotate(x[$w].value, r[$w].value))) for w ∈ 1:W]...))
    end
end

function rand_pcg_int32_quote(N, W)
    output = Expr(:tuple)
    Nreps, r = divrem(W, W64)
    r == 0 || throw("0 != $W % $W64 = $r.")
    q = quote
        $(Expr(:meta, :inline))
        prng = pointer(rng)
    end
    if Nreps > N
        NNrep, rr = divrem(Nreps, N)
        for n ∈ 1:N
            push!(q.args, quote
                $(Symbol(:state_, n)) = unsafe_load(prng, $n)
                $(Symbol(:increment_, n)) = unsafe_load(prng,$(N+n))
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
                    $state = vmuladd(multiplier, $state, $(Symbol(:increment_, n)))
                end)
                for w ∈ 1:2:W32
                    push!(output.args, :(VE(rotate($xorshifted[$w].value, $rot[$w].value))))
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
                $state = vmuladd(multiplier, $state, $(Symbol(:increment_, n)))
            end)
            for w ∈ 1:2:W32
                push!(output.args, :(VE(rotate($xorshifted[$w].value, $rot[$w].value))))
            end
        end
        for n ∈ 1:N
            push!(q.args, quote
                unsafe_store!(prng, $(Symbol(:state_, n)), $n)
            end)
        end
    else # Nreps <= N
        for n ∈ 1:Nreps
            state = Symbol(:state_, n)
            xorshifted = Symbol(:xorshifted_, n)
            rot = Symbol(:rot_, n)
            push!(q.args, quote
                $state = unsafe_load(prng, $n)
                unsafe_store!(prng, vmuladd(rng.multiplier, $state, unsafe_load(prng,$(N+n))), $n)
                $xorshifted = pirate_reinterpret(Vec{$W32,UInt32}, vright_bitshift(
                    vxor(
                        vright_bitshift($state, 18), $state
                    ), 27
                ))
                $rot = pirate_reinterpret(Vec{$W32,UInt32},vright_bitshift($state, 59))
            end)
            for w ∈ 1:2:W32
                push!(output.args, :(VE(rotate($xorshifted[$w].value, $rot[$w].value))))
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

@generated function Random.rand(rng::AbstractPCG_XSH_RR{N}, ::Type{Vec{W,UInt32}}) where {N,W}
    rand_pcg_int32_quote(N, W)
end
function rand_pcg_float_quote(N, W,::Type{Float32})
    output = Expr(:tuple)
    for w ∈ 1:W
        push!(output.args, :(VE(2f0 - mask(int[$w].value, Float32))))
    end
    quote
        int = $(rand_pcg_int32_quote(N, W))
        $output
    end
end
function rand_pcg_float_quote(N,W,::Type{Float64})
    output = Expr(:tuple)
    for w ∈ 1:W
        push!(output.args, :(VE(2.0 - mask(int[$w].value, Float64))))
    end
    quote
        int = pirate_reinterpret(Vec{$W,UInt64}, $(rand_pcg_int32_quote(N, W << 1)))
        $output
    end
end
@generated function Random.rand(rng::AbstractPCG_XSH_RR{N}, ::Type{Vec{W,T}}) where {N,W,T}
    rand_pcg_float_quote(N,W,T)
end
function subset_vec(name, W, offset = 0)
    Expr(:tuple, [:($name[$(offset+w)]) for w ∈ 1:W]...)
end
# @generated function Random.randexp(rng::AbstractPCG_XSH_RR{N}, ::Type{Vec{W,Float32}})
#     NW, r = divrem(W, W32)
#     output = Expr(:tuple)
#     q = quote
#         u = $(rand_pcg_float_quote(N,W,Float32))
#     end
#     for n ∈ 1:NW
#         e_n = Symbol(:e_,n)
#         push!(q.args,
#             :($e_n = SIMDPirates.vabs(SLEEFwrap.log_fast($(subset_vec(:u,W32,(n-1)*W32)))))
#         )
#         for w ∈ 1:W
#             push!(output.args, :($e_n[$w]))
#         end
#     end
#     if r > 0
#         e_n = Symbol(:e_,NW+1)
#         push!(q.args,
#             :($e_n = SIMDPirates.vabs(SLEEFwrap.log_fast($(subset_vec(:u,r,(NW-1)*W32)))))
#         )
#         for w ∈ 1:r
#             push!(output.args, :($e_n[$w]))
#         end
#     end
#     push!(q.args, output)
#     q
# end
# @generated function randmexp(rng::AbstractPCG_XSH_RR{N}, ::Type{Vec{W,Float32}})
#     NW, r = divrem(W, W32)
#     output = Expr(:tuple)
#     q = quote
#         u = $(rand_pcg_float32_quote(N,W))
#     end
#     for n ∈ 1:NW
#         e_n = Symbol(:e_,n)
#         push!(q.args,
#             :($e_n = SLEEFwrap.log_fast($(subset_vec(:u,W32,(n-1)*W32))))
#         )
#         for w ∈ 1:W
#             push!(output.args, :($e_n[$w]))
#         end
#     end
#     if r > 0
#         e_n = Symbol(:e_,NW+1)
#         push!(q.args,
#             :($e_n = SLEEFwrap.log_fast($(subset_vec(:u,r,NW*W32))))
#         )
#         for w ∈ 1:r
#             push!(output.args, :($e_n[$w]))
#         end
#     end
#     push!(q.args, output)
#     q
# end
@generated function Random.randexp(rng::AbstractPCG_XSH_RR{N}, ::Type{Vec{W,T}}) where {N,W,T}
    WT = TypeVectorWidth(T)
    NW, r = divrem(W, WT)
    output = Expr(:tuple)
    q = quote
        u = $(rand_pcg_float_quote(N,W,T))
    end
    for n ∈ 1:NW
        e_n = Symbol(:e_,n)
        push!(q.args,
            :($e_n = SIMDPirates.vabs(SLEEFwrap.log_fast($(subset_vec(:u,WT,(n-1)*WT)))))
        )
        for w ∈ 1:WT
            push!(output.args, :($e_n[$w]))
        end
    end
    if r > 0
        e_n = Symbol(:e_,NW+1)
        push!(q.args,
            :($e_n = SIMDPirates.vabs(SLEEFwrap.log_fast($(subset_vec(:u,r,NW*WT)))))
        )
        for w ∈ 1:r
            push!(output.args, :($e_n[$w]))
        end
    end
    push!(q.args, output)
    q
end
@generated function randmexp(rng::AbstractPCG_XSH_RR{N}, ::Type{Vec{W,T}}) where {N,W,T}
    WT = TypeVectorWidth(T)
    NW, r = divrem(W, WT)
    output = Expr(:tuple)
    q = quote
        u = $(rand_pcg_float_quote(N,W,T))
    end
    for n ∈ 1:NW
        e_n = Symbol(:e_,n)
        push!(q.args,
            :($e_n = SLEEFwrap.log_fast($(subset_vec(:u,WT,(n-1)*WT))))
        )
        for w ∈ 1:WT
            push!(output.args, :($e_n[$w]))
        end
    end
    if r > 0
        e_n = Symbol(:e_,NW+1)
        push!(q.args,
            :($e_n = SLEEFwrap.log_fast($(subset_vec(:u,r,NW*WT))))
        )
        for w ∈ 1:r
            push!(output.args, :($e_n[$w]))
        end
    end
    push!(q.args, output)
    q
end
@generated function Random.randn(rng::AbstractPCG_XSH_RR{N}, ::Type{Vec{W,T}}) where {N,W,T}
    WT = TypeVectorWidth(T)
    NW, r = divrem(W >> 1, WT)
    # workaround
    splitsincos = WT * sizeof(T) < 64

    output = Expr(:tuple)
    q = quote
        u = $(rand_pcg_float_quote(N,W,T))
    end
    for n ∈ 1:NW
        u1_n = Symbol(:u1_, n)
        u2_n = Symbol(:u2_, n)
        # get the vectors u_1 and u_2
        push!(q.args, quote
            $u1_n = SLEEFwrap.log_fast($(subset_vec(:u,WT,(n-1)*2WT)))
            $u2_n =                    $(subset_vec(:u,WT,(n-1)*2WT + WT))
            $u1_n = SIMDPirates.vsqrt( SIMDPirates.vabs( SIMDPirates.vadd($u1_n, $u1_n) ) )
            $u2_n = SIMDPirates.vadd($u2_n, $u2_n)
        end)
        s_n = Symbol(:s_, n)
        c_n = Symbol(:c_, n)
        # workaround for https://github.com/JuliaLang/julia/issues/30426
        if splitsincos
            push!(q.args, :(($s_n, $c_n) = SLEEFwrap.sincospi_fast($u2_n)) )
        else
            sc_n = Symbol(:sc_, n)
            push!(q.args,  quote
                $sc_n = SLEEFwrap.sincospi_fast($u2_n)
                $s_n = $(subset_vec(sc_n, WT, 0))
                $c_n = $(subset_vec(sc_n, WT, WT))
            end)
        end
        z1_n = Symbol(:z1_,n)
        z2_n = Symbol(:z2_,n)
        push!(q.args, quote
            $z1_n = SIMDPirates.extract_data(SIMDPirates.vmul($u1_n, $s_n))
            $z2_n = SIMDPirates.extract_data(SIMDPirates.vmul($u1_n, $c_n))
        end )
        for w ∈ 1:WT
            push!(output.args, :($z1_n[$w]))
        end
        for w ∈ 1:WT
            push!(output.args, :($z2_n[$w]))
        end
    end
    if r > 0
        u1_n = Symbol(:u1_, NW+1)
        u2_n = Symbol(:u2_, NW+1)
        # get the vectors u_1 and u_2
        push!(q.args, quote
            $u1_n = SLEEFwrap.log_fast($(subset_vec(:u,r,NW*2WT)))
            $u2_n =                    $(subset_vec(:u,r,NW*2WT+r))
            $u1_n = SIMDPirates.vsqrt( SIMDPirates.vabs( SIMDPirates.vadd($u1_n, $u1_n) ) )
            $u2_n = SIMDPirates.vadd($u2_n, $u2_n)
        end)
        s_n = Symbol(:s_, NW+1)
        c_n = Symbol(:c_, NW+1)
        # workaround for https://github.com/JuliaLang/julia/issues/30426
        # AFAIK r * sizeof(T) < 64 for all supported use cases
        if r * sizeof(T) < 64
            push!(q.args, :(($s_n, $c_n) = SLEEFwrap.sincospi_fast($u2_n)) )
        else
            sc_n = Symbol(:sc_, NW+1)
            push!(q.args,  quote
                $sc_n = SLEEFwrap.sincospi_fast($u2_n)
                $s_n = $(subset_vec(sc_n, r, 0))
                $c_n = $(subset_vec(sc_n, r, r))
            end)
        end
        z1_n = Symbol(:z1_,NW+1)
        z2_n = Symbol(:z2_,NW+1)
        push!(q.args, quote
            $z1_n = SIMDPirates.extract_data(SIMDPirates.vmul($u1_n, $s_n))
            $z2_n = SIMDPirates.extract_data(SIMDPirates.vmul($u1_n, $c_n))
        end )
        for w ∈ 1:r
            push!(output.args, :($z1_n[$w]))
        end
        for w ∈ 1:r
            push!(output.args, :($z2_n[$w]))
        end
    end
    push!(q.args, output)
    q
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

@generated function Random.rand!(rng::AbstractPCG_XSH_RR{N}, x::AbstractArray{T}) where {N,T <: Real}
    WT = TypeVectorWidth(T)
    Nhalf = N >> 1
    NWT = Nhalf*WT
    # float_q = rand_pcg_float_quote(N,NWT,T)
    float_q = :(rand(rng, Vec{$NWT,$T}))
    store_expr = quote end
    for n ∈ 0:Nhalf-1
        push!(store_expr.args, :(SIMDPirates.vstore($(subset_vec(:u, WT, n*WT)), ptr_x, i + $(n*WT))))
    end
    unrolled_rand_quote(NWT, float_q, store_expr)
end
@generated function Random.randexp!(rng::AbstractPCG_XSH_RR{N}, x::AbstractArray{T}) where {N,T <: Real}
    WT = TypeVectorWidth(T)
    Nhalf = N >> 1
    NWT = Nhalf*WT
    store_expr = quote end
    for n ∈ 0:Nhalf
        push!(store_expr.args, :(SIMDPirates.vstore($(subset_vec(:u, WT, n*WT)), ptr_x, i + $(n*WT))))
    end
    unrolled_rand_quote(NWT, :(randexp(rng, Vec{$NWT,$T})), store_expr)
end
@generated function Random.randn!(rng::AbstractPCG_XSH_RR{N}, x::AbstractArray{T}) where {N,T <: Real}
    WT = TypeVectorWidth(T)
    Nhalf = N >> 1
    NWT = Nhalf*WT
    store_expr = quote end
    for n ∈ 0:Nhalf-1
        push!(store_expr.args, :(SIMDPirates.vstore($(subset_vec(:u, WT, n*WT)), ptr_x, i + $(n*WT))))
    end
    unrolled_rand_quote(NWT, :(randn(rng, Vec{$NWT,$T})), store_expr)
end






function rand_UInt_quote(N, ::Type{UT}) where UT
    L64 = 2 * sizeof(UT) ÷ sizeof(UInt64)
    L64_2 = 2L64
    W_ratio = W64 ÷ L64
    increment =   N * W_ratio + 1
    multiplier = 2N * W_ratio + 1
    output = Expr(:tuple, [:( VE(rotate(xorshifted[$i].value, rot[$i].value)) ) for i ∈ 1:2:L64_2]...)
    quote
        prng = Base.unsafe_convert(Ptr{Vec{$L64,UInt64}}, pointer_from_objref(rng))
        state = unsafe_load(prng)
        xorshifted = pirate_reinterpret(Vec{$L64_2,UInt32}, vright_bitshift(
            vxor(
                vright_bitshift(state, 18), state
            ), 27
        ))
        rot = pirate_reinterpret(Vec{$L64_2,UInt32},vright_bitshift(state, 59))
        unsafe_store!(prng, vmuladd(unsafe_load(prng, $multiplier), state, unsafe_load(prng, $increment)))

        pirate_reinterpret(Vec{1,$UT}, $output)[1].value
    end
end

@generated function Random.rand(rng::PCGCore{N}, ::Type{UT}) where {UT <: Union{UInt32,UInt64}, N}
    rand_UInt_quote(N, UT)
end
@generated function Random.rand(rng::PCGCore{N}, ::Type{Float64}) where N
    quote
        u64 = $(rand_UInt_quote(N, UInt64))
        2 - mask(u64, Float64)
    end
end
@generated function Random.rand(rng::PCGCore{N}, ::Type{Float32}) where N
    quote
        u32 = $(rand_UInt_quote(N, UInt32))
        2 - mask(u32, Float32)
    end
end
@generated function Random.rand(rng::PCGCore{N}, ::Random.UInt52{UInt64}) where N
    quote
        u64 = $(rand_UInt_quote(N, UInt64))
        u64 & 0x000fffffffffffff
    end
end
@generated function Random.rand(rng::PCGCore{N}, ::Random.UInt23{UInt32}) where N
    quote
        u32 = $(rand_UInt_quote(N, UInt32))
        U32 & 0x007fffff
    end
end










@generated function reset_rand64_state!(rng::PCG_cached_64{N,W64_}) where {N,W64_}
    # W32_ = N * W64
    W = N * W64 >> 1
    nWs, r = divrem(W64_, W)
    V = Vec{W,Float64}
    r == 0 || throw("$W64_ % $W = r == $r != 0")
    quote
        $(Expr(:meta, :noinline))
        rng.uniformcount64 = 1
        ptr_rng = Base.unsafe_convert(Ptr{$V}, pointer_from_objref(rng) + $(cache64_offset(N)))
        for n ∈ 1:$nWs
            u = rand(rng, $V)
            unsafe_store!(ptr_rng, u, n)
        end
        return @inbounds rng.uniform64[1].value
    end
end
Random.rand(rng::PCG_cached_64, ::Type{Float64}) = rand(rng)
function Random.rand(rng::PCG_cached_64{N,W64_}) where {N,W64_}
    if rng.uniformcount64 < W64_
        rng.uniformcount64 += 1
        return @inbounds rng.uniform64[rng.uniformcount64].value
    else
        return reset_rand64_state!(rng)
    end
end
@generated function reset_rand32_state!(rng::PCG_cached_32{N,W32_}) where {N,W32_}
    # W32_ = N * W64
    W = N * W32 >> 1
    nWs, r = divrem(W32_, W)
    V = Vec{W, Float32}
    r == 0 || throw("$W64_ % $W = r == $r != 0")
    quote
        $(Expr(:meta, :noinline))
        rng.uniformcount32 = 1
        ptr_rng = Base.unsafe_convert(Ptr{$V}, pointer_from_objref(rng) + $(cache32_offset(rng)))
        for n ∈ 1:$nWs
            u = rand(rng, $V)
            unsafe_store!(ptr_rng, u, n)
        end
        # return unsafe_load(Base.unsafe_convert(Ptr))
        return @inbounds rng.uniform32[1].value
    end
end
@inline Random.rand(rng::PCGcached32) = rand(rng, Float32)
function Random.rand(rng::PCG_cached_32{N,W32_}, ::Type{Float32}) where {N,W32_}
    if rng.uniformcount32 < W32_
        rng.uniformcount32 += 1
        return @inbounds rng.uniform32[rng.uniformcount32].value
    else
        return reset_rand32_state!(rng)
    end
end
@generated function reset_randexp64_state!(rng::PCG_cached_64{N,W64_}) where {N,W64_}
    # W32_ = N * W64
    W = N * W64 >> 1
    nWs, r = divrem(W64_, W)
    V = Vec{W,Float64}
    r == 0 || throw("$W64_ % $W = r == $r != 0")
    quote
        $(Expr(:meta, :noinline))
        rng.exponentialcount64 = 1
        ptr_rng = Base.unsafe_convert(Ptr{$V}, pointer_from_objref(rng) + $(cache64_offset(N)+W64_*sizeof(Float64)))
        for n ∈ 1:$nWs
            re = randexp(rng, $V)
            unsafe_store!(ptr_rng, re, n)
        end
        return unsafe_load(Base.unsafe_convert(Ptr{Float64}, ptr_rng))
    end
end
Random.randexp(rng::PCG_cached_64, ::Type{Float64}) = randexp(rng)
function Random.randexp(rng::PCG_cached_64{N,W64_}) where {N,W64_}
    if rng.exponentialcount64 < W64
        rng.exponentialcount64 += 1
        return @inbounds rng.exponential64[rng.exponentialcount64].value
    else
        return reset_randexp64_state!(rng)
    end
end
@generated function reset_randexp32_state!(rng::PCG_cached_32{N,W32_}) where {N,W32_}
    # W32_ = N * W64
    W = N * W32 >> 1
    nWs, r = divrem(W32_, W)
    V = Vec{W,Float32}
    r == 0 || throw("$W64_ % $W = r == $r != 0")
    quote
        $(Expr(:meta, :noinline))
        rng.normalcount32 = 1
        ptr_rng = Base.unsafe_convert(Ptr{$V}, pointer_from_objref(rng) + $(cache32_offset(rng)+W32_*sizeof(Float32)))
        for n ∈ 1:$nWs
            rn = randexp(rng, $V)
            unsafe_store!(ptr_rng, rn, n)
        end
        return unsafe_load(Base.unsafe_convert(Ptr{Float32}, ptr_rng))
    end
end
@inline Random.randexp(rng::PCGcached32) = randexp(rng, Float32)
function Random.randexp(rng::PCG_cached_32{N,W32_}, ::Type{Float32}) where {N,W32_}
    if rng.exponentialcount32 < W32_
        rng.exponentialcount32 += 1
        return @inbounds rng.exponential32[rng.exponentialcount32].value
    else
        return reset_randexp32_state!(rng)
    end
end
@generated function reset_randn64_state!(rng::PCG_cached_64{N,W64_}) where {N,W64_}
    # W32_ = N * W64
    W = N * W64 >> 1
    nWs, r = divrem(W64_, W)
    V = Vec{W,Float64}
    r == 0 || throw("$W64_ % $W = r == $r != 0")
    quote
        $(Expr(:meta, :noinline))
        rng.normalcount64 = 1
        ptr_rng = Base.unsafe_convert(Ptr{$V}, pointer_from_objref(rng) + $(cache64_offset(N)+2W64_*sizeof(Float64)))
        for n ∈ 1:$nWs
            rn = randn(rng, $V)
            unsafe_store!(ptr_rng, rn, n)
        end
        return unsafe_load(Base.unsafe_convert(Ptr{Float64}, ptr_rng))
    end
end
Random.randn(rng::PCG_cached_64, ::Type{Float64}) = randn(rng)
function Random.randn(rng::PCG_cached_64{N,W64_}) where {N,W64_}
    nc = rng.normalcount64
    if rng.normalcount64 < W64_
        rng.normalcount64 += 1
        return @inbounds rng.normal64[rng.normalcount64].value
    else
        return reset_randn64_state!(rng)
    end
end
@generated function reset_randn32_state!(rng::PCG_cached_32{N,W32_}) where {N,W32_}
    # W32_ = N * W64
    W = N * W32 >> 1
    nWs, r = divrem(W32_, W)
    V = Vec{W,Float32}
    r == 0 || throw("$W64_ % $W = r == $r != 0")
    quote
        $(Expr(:meta, :noinline))
        rng.normalcount32 = 1
        ptr_rng = Base.unsafe_convert(Ptr{$V}, pointer_from_objref(rng) + $(cache32_offset(rng)+2W32_*sizeof(Float32)))
        for n ∈ 1:$nWs
            rn = randn(rng, $V)
            unsafe_store!(ptr_rng, rn, n)
        end
        return unsafe_load(Base.unsafe_convert(Ptr{Float32}, ptr_rng))
    end
end
@inline Random.randn(rng::PCGcached32) = randn(rng, Float32)
function Random.randn(rng::PCG_cached_32{N,W32_}, ::Type{Float32}) where {N,W32_}
    if rng.normalcount32 < W32_
        rng.normalcount32 += 1
        return @inbounds rng.normal32[rng.normalcount32].value
    else
        return reset_randn32_state!(rng)
    end
end
