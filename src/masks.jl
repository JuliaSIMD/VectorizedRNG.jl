
"""
setbits(x::Unsigned, y::Unsigned, mask::Unsigned)

If you have AVX512, setbits of vector-arguments will select bits according to mask `m`, selecting from `y` if 0 and from `x` if `1`.
For scalar arguments, or vector arguments without AVX512, `setbits` requires the additional restrictions on `y` that all bits for
which `m` is 1, `y` must be 0.
That is for scalar arguments or vector arguments without AVX512, it requires the restriction that
((y ⊻ m) & m) == m
"""
@inline setbits(x, y, m) = (x & m) | y
@inline function setbits(
  x::Vec{W,U},
  y,
  m,
  ::True
) where {W,U<:Union{UInt32,UInt64}}
  VectorizationBase.vpternlog(
    vbroadcast(Val{W}(), y),
    x,
    vbroadcast(Val{W}(), m),
    Val{216}()
  )
end
@inline function setbits(
  x::VecUnroll{N,W,U},
  y,
  m,
  ::True
) where {N,W,U<:Union{UInt32,UInt64}}
  VectorizationBase.VecUnroll(VectorizationBase.fmap(setbits, data(x), y, m))
end
@inline setbits(x, y, m, ::False) = ((x & m) | y)
@inline setbits(x::VectorizationBase.AbstractSIMD, y, m) =
  setbits(x, y, m, VectorizationBase.has_feature(Val(:x86_64_avx512f)))

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
@inline floatbitmask(x, ::Type{Float64}) = reinterpret(
  Float64,
  setbits(reinterpret(UInt64, x), 0x3ff0000000000000, 0x000fffffffffffff)
)
@inline floatbitmask(x, ::Type{Float32}) =
  reinterpret(Float32, setbits(reinterpret(UInt32, x), 0x3f800000, 0x007fffff))
