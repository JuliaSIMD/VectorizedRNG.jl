module VectorizedRNG

using VectorizationBase, SIMDPirates, SLEEF, Random
using VectorizationBase: VE, REGISTER_SIZE
using SIMDPirates: pirate_reinterpret, vxor, vor, vand, vright_bitshift,
                        vbroadcast, vadd, vmul, vsub, vabs, vsqrt,
                        extract_data, vstore!

const W64 = REGISTER_SIZE >> 3
const W32 = REGISTER_SIZE >> 2
const W16 = REGISTER_SIZE >> 1




include("multipliers.jl")
include("PCG.jl")
include("random_distributions.jl")

const GLOBAL_vPCG = PCG{4}()

end # module
