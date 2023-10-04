module VectorizedRNG
if isdefined(Base, :Experimental) &&
   isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end

using VectorizationBase, Random, UnPack
using VectorizationBase:
  simd_integer_register_size,
  gep,
  _Vec,
  ifelse,
  VecUnroll,
  AbstractSIMD,
  rotate_right,
  vadd,
  vsub,
  zero_offsets,
  vfmadd,
  vfmsub,
  vfnmadd,
  shufflevector,
  register_size,
  cache_linesize,
  StaticInt,
  pick_vector_width,
  data
using SLEEFPirates

using Distributed: myid

if !isdefined(Base, :get_extension)
  using Requires
end

export local_rng, rand!, randn!#, randexp, randexp!

abstract type AbstractVRNG{N} <: Random.AbstractRNG end
abstract type AbstractState{N,W} end

@inline vloadu(ptr::Ptr) =
  VectorizationBase.__vload(ptr, VectorizationBase.False(), register_size())
@inline vloadu(ptr::Ptr, i) =
  VectorizationBase.__vload(ptr, i, VectorizationBase.False(), register_size())

const ALIGN = VectorizationBase.False()

@inline vloada(ptr::Ptr) =
  VectorizationBase.__vload(ptr, ALIGN, register_size())
@inline vloada(ptr::Ptr, i) =
  VectorizationBase.__vload(ptr, i, ALIGN, register_size())
@inline vloada(ptr::Ptr, i, m) =
  VectorizationBase.__vload(ptr, i, m, ALIGN, register_size())
@inline vstorea!(ptr::Ptr, v) = VectorizationBase.__vstore!(
  ptr,
  v,
  ALIGN,
  VectorizationBase.False(),
  VectorizationBase.False(),
  register_size()
)
@inline vstorea!(ptr::Ptr, v, i) = VectorizationBase.__vstore!(
  ptr,
  v,
  i,
  ALIGN,
  VectorizationBase.False(),
  VectorizationBase.False(),
  register_size()
)
@inline vstorea!(ptr::Ptr, v, i, m) = VectorizationBase.__vstore!(
  ptr,
  v,
  i,
  m,
  ALIGN,
  VectorizationBase.False(),
  VectorizationBase.False(),
  register_size()
)
@inline vstoreu!(ptr::Ptr, v, i) = VectorizationBase.__vstore!(
  ptr,
  v,
  i,
  VectorizationBase.False(),
  VectorizationBase.False(),
  VectorizationBase.False(),
  register_size()
)
@inline vstoreu!(ptr::Ptr, v) = VectorizationBase.__vstore!(
  ptr,
  v,
  VectorizationBase.False(),
  VectorizationBase.False(),
  VectorizationBase.False(),
  register_size()
)

include("masks.jl")
include("api.jl")
include("special_approximations.jl")
include("xoshiro.jl")
# const GLOBAL_vPCGs = Ref{Ptr{UInt64}}()

const GLOBAL_vRNGs = Ref{Ptr{UInt64}}()

const RNG_MEM_SIZE = (5(simd_integer_register_size() * XREGISTERS + 2048 * 3))
local_rng(i) = Xoshiro{XREGISTERS}(i * (RNG_MEM_SIZE) + GLOBAL_vRNGs[])
local_rng() = local_rng(Base.Threads.threadid() - 1)

function __init()
  nthreads = Base.Threads.nthreads()
  GLOBAL_vRNGs[] =
    ptr = VectorizationBase.valloc((RNG_MEM_SIZE ÷ 8) * nthreads, UInt64)
  nstreams = XREGISTERS * nthreads * simd_integer_register_size()
  initXoshiro!(ptr, nstreams)
  for tid ∈ 0:nthreads-1
    rng = local_rng(tid)
    setrandu64counter!(rng, 0x00)
    # setrandn32counter!(rng, 0x00)
    setrand64counter!(rng, 0x00)
    setrandn64counter!(rng, 0x00)
  end
end
function __init__()
  @static if !isdefined(Base, :get_extension)
    @require StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c" begin
      include("../ext/VectorizedRNGStaticArraysExt.jl")
    end
  end
  ccall(:jl_generating_output, Cint, ()) == 1 && return
  __init()
end

end # module
