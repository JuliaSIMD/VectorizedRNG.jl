module VectorizedRNG

using VectorizationBase, SIMDPirates, SLEEFPirates, Random
using VectorizationBase: VE, REGISTER_SIZE
using SIMDPirates: pirate_reinterpret, vxor, vor, vand, vuright_bitshift,
                        vbroadcast, vadd, vmul, vsub, vabs, vsqrt,
                        extract_data, vstore!

using Distributed: myid

const W64 = REGISTER_SIZE >> 3
const W32 = REGISTER_SIZE >> 2
const W16 = REGISTER_SIZE >> 1




include("multipliers.jl")
include("PCG.jl")
include("random_distributions.jl")

const GLOBAL_vPCG = PCG{4}(undef)
include("precompile.jl")
_precompile_()

function __init__()
    random_init_pcg!(GLOBAL_vPCG, myid() - 1)
    _precompile_()
end

    
end # module
