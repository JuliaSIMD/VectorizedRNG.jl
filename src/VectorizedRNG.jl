module VectorizedRNG

using VectorizationBase, Random, UnPack
using VectorizationBase: simd_integer_register_size, gep, _Vec, ifelse, VecUnroll, AbstractSIMD,
    rotate_right, vadd, vsub, zero_offsets, vfmadd, vfmsub, vfnmadd, shufflevector, register_size,
    cache_linesize, StaticInt, pick_vector_width, data
using SLEEFPirates

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


const RNG_MEM_SIZE = (5(simd_integer_register_size()*XREGISTERS + 2048*3))
local_rng(i) = Xoshift{XREGISTERS}(i*(RNG_MEM_SIZE) + GLOBAL_vRNGs[])
local_rng() = local_rng(Base.Threads.threadid() - 1)

# include("precompile.jl")
# _precompile_()

function __init__()
    # ccall(:jl_generating_output, Cint, ()) == 1 && return
  nthreads = Base.Threads.nthreads()
  GLOBAL_vRNGs[] = ptr = VectorizationBase.valloc((RNG_MEM_SIZE ÷ 8)*nthreads, UInt64)
  nstreams = XREGISTERS * nthreads * simd_integer_register_size()
  initXoshift!(ptr, nstreams)
  for tid ∈ 0:nthreads-1
    rng = local_rng(tid)
    setrandu64counter!(rng, 0x00)
    # setrandn32counter!(rng, 0x00)
    setrand64counter!(rng, 0x00)
    setrandn64counter!(rng, 0x00)
  end
end

    
end # module
