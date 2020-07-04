module VectorizedRNG

using VectorizationBase, SIMDPirates, Random
using VectorizationBase: VE, REGISTER_SIZE, gep, _Vec
using SIMDPirates: vreinterpret, vxor, vor, vand, vuright_bitshift,
                        vbroadcast, vadd, vmul, vsub, vabs, vsqrt,
                        extract_data, vcopysign, vleft_bitshift, vuright_bitshift

using Distributed: myid

export local_rng, rand!, randn!#, randexp, randexp!

abstract type AbstractVRNG{N} <: Random.AbstractRNG end
abstract type AbstractState{N,W} end

const W64 = REGISTER_SIZE >> 3
const W32 = REGISTER_SIZE >> 2
const W16 = REGISTER_SIZE >> 1

@inline rotate(x::UInt32, r) = x >>> r | x << (0x00000020 - r)
@inline function rotate(x::_Vec{W,UInt32}, r::_Vec{W,T}) where {W,T}
    xshiftright = SIMDPirates.vuright_bitshift(x, r)
    xshiftleft = SIMDPirates.vleft_bitshift(x, SIMDPirates.vsub(SIMDPirates.vbroadcast(_Vec{W,T}, 0x00000020), r))
    SIMDPirates.vor(xshiftright, xshiftleft)
end
@inline rotate(x::UInt64, r) = x >>> r | x << (0x0000000000000040 - r)
@inline function rotate(x::_Vec{W,UInt64}, r::_Vec{W,T}) where {W,T}
    xshiftright = SIMDPirates.vuright_bitshift(x, r)
    xshiftleft = SIMDPirates.vleft_bitshift(x, SIMDPirates.vsub(SIMDPirates.vbroadcast(_Vec{W,T}, 0x0000000000000040), r))
    SIMDPirates.vor(xshiftright, xshiftleft)
end
@inline rotate(x::_Vec{W,U}, r::U) where {W,U} = rotate(x, vbroadcast(_Vec{W,U}, r))

include("masks.jl")
include("api.jl")
include("special_approximations.jl")
include("xoshiro.jl")
# const GLOBAL_vPCGs = Ref{Ptr{UInt64}}()


const GLOBAL_vRNGs = Ref{Ptr{UInt64}}()

local_rng(i) = Xoshift{XREGISTERS}(i*4REGISTER_SIZE*XREGISTERS + GLOBAL_vRNGs[])
local_rng() = local_rng(Base.Threads.threadid() - 1)

# include("precompile.jl")
# _precompile_()

function __init__()
    nthreads = Base.Threads.nthreads()
    nstreams = XREGISTERS * nthreads * W64
    GLOBAL_vRNGs[] = ptr = VectorizationBase.valloc(4nstreams, UInt64)
    initXoshift!(ptr, nstreams)
end

    
end # module
