function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{Core.kwftype(typeof(VectorizedRNG.rand_pcg_float_quote)),NamedTuple{(:uload,),Tuple{Bool}},typeof(VectorizedRNG.rand_pcg_float_quote),Int64,Int64,Int64,Type{Float64},VectorizedRNG.PCG_Algorithm})
    precompile(Tuple{typeof(VectorizedRNG.mask_expr),Int64,Type{T} where T,Type{Float64},Expr})
    precompile(Tuple{typeof(VectorizedRNG.rand_pcg_float_quote),Int64,Int64,Int64,Type{Float64},VectorizedRNG.PCG_Algorithm,Symbol,Symbol})
    precompile(Tuple{typeof(VectorizedRNG.rand_pcg_float_quote),Int64,Int64,Type{Float64},VectorizedRNG.PCG_Algorithm})
    precompile(Tuple{typeof(VectorizedRNG.randn_quote),Int64,Int64,Int64,Type{T} where T,VectorizedRNG.PCG_Algorithm})
    precompile(Tuple{typeof(VectorizedRNG.random_init_pcg!),VectorizedRNG.PtrPCG{4},Int64})
end
