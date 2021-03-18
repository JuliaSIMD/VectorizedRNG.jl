module VectorizedRNG

using VectorizationBase, Random, UnPack
using VectorizationBase: simd_integer_register_size, gep, _Vec, ifelse, VecUnroll, AbstractSIMD,
    rotate_right, vadd, vsub, zero_offsets, vfmadd, vfmsub, vfnmadd, shufflevector, register_size,
    cache_linesize, StaticInt, pick_vector_width, data

using Distributed: myid

export local_rng, rand!, randn!#, randexp, randexp!

abstract type AbstractVRNG{N} <: Random.AbstractRNG end
abstract type AbstractState{N,W} end

@inline vloadu(ptr::Ptr) = VectorizationBase.__vload(ptr, VectorizationBase.False(), register_size())
@inline vloadu(ptr::Ptr, i) = VectorizationBase.__vload(ptr, i, VectorizationBase.False(), register_size())
@inline vloada(ptr::Ptr) = VectorizationBase.__vload(ptr, VectorizationBase.True(), register_size())
@inline vloada(ptr::Ptr, i) = VectorizationBase.__vload(ptr, i, VectorizationBase.True(), register_size())
@inline vloada(ptr::Ptr, i, m) = VectorizationBase.__vload(ptr, i, m, VectorizationBase.True(), register_size())
@inline vstorea!(ptr::Ptr, v) = VectorizationBase.__vstore!(ptr, v, VectorizationBase.True(), VectorizationBase.False(), VectorizationBase.False(), register_size())
@inline vstorea!(ptr::Ptr, v, i) = VectorizationBase.__vstore!(ptr, v, i, VectorizationBase.True(), VectorizationBase.False(), VectorizationBase.False(), register_size())
@inline vstorea!(ptr::Ptr, v, i, m) = VectorizationBase.__vstore!(ptr, v, i, m, VectorizationBase.True(), VectorizationBase.False(), VectorizationBase.False(), register_size())
@inline vstoreu!(ptr::Ptr, v, i) = VectorizationBase.__vstore!(ptr, v, i, VectorizationBase.False(), VectorizationBase.False(), VectorizationBase.False(), register_size())
@inline vstoreu!(ptr::Ptr, v) = VectorizationBase.__vstore!(ptr, v, VectorizationBase.False(), VectorizationBase.False(), VectorizationBase.False(), register_size())



include("masks.jl")
include("api.jl")
include("special_approximations.jl")
include("xoshiro.jl")
# const GLOBAL_vPCGs = Ref{Ptr{UInt64}}()

const GLOBAL_vRNGs = Ref{Ptr{UInt64}}()

local_rng(i) = Xoshift{XREGISTERS}(i*4simd_integer_register_size()*XREGISTERS + GLOBAL_vRNGs[])
local_rng() = local_rng(Base.Threads.threadid() - 1)

# include("precompile.jl")
# _precompile_()

# const RANDBUFFER32 = Float32[]
# const RANDNBUFFER32 = Float32[]
# const RANDBUFFER64 = Float64[]
# const RANDNBUFFER64 = Float64[]
# const RANDBUFFERCOUNTER = UInt8[]
# const RANDNBUFFER32COUNTER = UInt8[]
# const RANDBUFFER64COUNTER = UInt8[]
# const RANDNBUFFER64COUNTER = UInt8[]



function __init__()
    # ccall(:jl_generating_output, Cint, ()) == 1 && return
    nthreads = Base.Threads.nthreads()
    nstreams = XREGISTERS * nthreads * simd_integer_register_size()
    GLOBAL_vRNGs[] = ptr = VectorizationBase.valloc(5nstreams + 256 * 3nthreads, UInt64)
    initXoshift!(ptr, nstreams)

    for tid ∈ 0:nthreads-1
        rng = local_rng(tid)
        setrandu64counter!(rng, 0x00)
        # setrandn32counter!(rng, 0x00)
        setrand64counter!(rng, 0x00)
        setrandn64counter!(rng, 0x00)
    end
    # resize!(RANDBUFFER32, 256nthreads)
    # resize!(RANDNBUFFER32, 256nthreads)
    # resize!(RANDBUFFER64, 256nthreads)
    # resize!(RANDNBUFFER64, 256nthreads)
    
    # resize!(RANDBUFFERCOUNTER, VectorizationBase.CACHELINE_SIZE*nthreads); fill!(RANDBUFFERCOUNTER, 0)
    # resize!(RANDNBUFFER32COUNTER, VectorizationBase.CACHELINE_SIZE*nthreads)
    # resize!(RANDBUFFER64COUNTER, VectorizationBase.CACHELINE_SIZE*nthreads)
    # resize!(RANDNBUFFER64COUNTER, VectorizationBase.CACHELINE_SIZE*nthreads)
end

    
end # module
