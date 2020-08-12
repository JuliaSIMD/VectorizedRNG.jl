module VectorizedRNG

using VectorizationBase, SIMDPirates, Random, UnPack
using VectorizationBase: VE, REGISTER_SIZE, gep, _Vec
using SIMDPirates: vreinterpret, vbroadcast, vadd, vmul, vsub, vabs, vsqrt,
                        extract_data, vcopysign, rotate_right

using Distributed: myid

export local_rng, rand!, randn!#, randexp, randexp!

abstract type AbstractVRNG{N} <: Random.AbstractRNG end
abstract type AbstractState{N,W} end

const W64 = REGISTER_SIZE >> 3
const W32 = REGISTER_SIZE >> 2
const W16 = REGISTER_SIZE >> 1

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
