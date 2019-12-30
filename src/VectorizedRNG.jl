module VectorizedRNG

using VectorizationBase, SIMDPirates, SLEEFPirates, Random
using VectorizationBase: VE, REGISTER_SIZE
using SIMDPirates: pirate_reinterpret, vxor, vor, vand, vuright_bitshift,
                        vbroadcast, vadd, vmul, vsub, vabs, vsqrt,
                        extract_data, vstore!

using Distributed: myid

export local_pcg, rand!, randn!, randexp!

const W64 = REGISTER_SIZE >> 3
const W32 = REGISTER_SIZE >> 2
const W16 = REGISTER_SIZE >> 1

include("multipliers.jl")
include("PCG.jl")
include("random_distributions.jl")

const GLOBAL_vPCGs = Ref{Ptr{UInt64}}()

local_pcg(i) = PtrPCG{4}(i*9REGISTER_SIZE + GLOBAL_vPCGs[])
local_pcg() = local_pcg(Base.Threads.threadid() - 1)

include("precompile.jl")
_precompile_()

function __init__()
    nthreads = Base.Threads.nthreads()
    GLOBAL_vPCGs[] = VectorizationBase.valloc(9W64*nthreads, UInt64)
    id = myid() - 1
    for t ∈ 0:nthreads-1
        random_init_pcg!(local_pcg(t), myid() - 1)
    end
end

    
end # module
