module VectorizedRNG

using VectorizationBase, SIMDPirates, SLEEFwrap, Random
using VectorizationBase: VE, REGISTER_SIZE
using SIMDPirates: pirate_reinterpret,
                            vxor, vright_bitshift

include("PCG.jl")

end # module
