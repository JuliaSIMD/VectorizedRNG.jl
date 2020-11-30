# VectorizedRNG

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/VectorizedRNG.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://chriselrod.github.io/VectorizedRNG.jl/latest)
[![Build Status](https://travis-ci.com/chriselrod/VectorizedRNG.jl.svg?branch=master)](https://travis-ci.com/chriselrod/VectorizedRNG.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/chriselrod/VectorizedRNG.jl?svg=true)](https://ci.appveyor.com/project/chriselrod/VectorizedRNG-jl)
[![Codecov](https://codecov.io/gh/chriselrod/VectorizedRNG.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chriselrod/VectorizedRNG.jl)


This library provides a vectorized Xoshiro256++ random number generator. The larger the host computers SIMD vector width, the better they will perform. On a machine with AVX-512, they are faster than [SIMD-oriented Fast Mersenne Twister (SFMT) ](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/). Base Julia uses dSFMT,  which in a few tests appears to outperform this library on avx2 systems in generating uniformly distributed random numbers.

You can get a thread-local instance of the `Xoshiro` generator with `local_rng()`. Each parallel stream jumps ahead `2^128` samples, which should be more than enough samples per stream for any real calculation. Each thread gets 8 parallel streams with AVX, or 16 with AVX512, allowing there to be up to `2^125` or `2^124` threads with AVX512.

Testing on an old haswell machine (AVX2-only):
```julia
julia> using BenchmarkTools, Random, VectorizedRNG

julia> x = Vector{Float64}(undef, 1024);

julia> @benchmark randn!($x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     7.235 μs (0.00% GC)
  median time:      7.900 μs (0.00% GC)
  mean time:        8.034 μs (0.00% GC)
  maximum time:     233.290 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     5
 
julia> @benchmark randn!(local_rng(), $x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     3.744 μs (0.00% GC)
  median time:      4.156 μs (0.00% GC)
  mean time:        4.137 μs (0.00% GC)
  maximum time:     59.169 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     8
```
The performance advantage is thanks primarily to a fast SIMD [Box-Muller](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform) implementation; `randn(::MersenneTwister)` uses the ziggurat algorithm, which is more efficient for scalars. Performance is closer when only comparing random-uniform generation:
```julia
julia> @benchmark rand!($x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     791.047 ns (0.00% GC)
  median time:      904.541 ns (0.00% GC)
  mean time:        915.753 ns (0.00% GC)
  maximum time:     13.978 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     85
 
julia> @benchmark rand!(local_rng(), $x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     513.000 ns (0.00% GC)
  median time:      568.578 ns (0.00% GC)
  mean time:        571.597 ns (0.00% GC)
  maximum time:     4.706 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     192
```
This library shines on a system with AVX512:
```julia
julia> using BenchmarkTools, Random, VectorizedRNG

julia> x = Vector{Float64}(undef, 1024);

julia> @benchmark randn!($x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     1.676 μs (0.00% GC)
  median time:      1.798 μs (0.00% GC)
  mean time:        1.883 μs (0.00% GC)
  maximum time:     5.769 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     10

julia> @benchmark randn!(local_rng(), $x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     854.446 ns (0.00% GC)
  median time:      962.369 ns (0.00% GC)
  mean time:        991.798 ns (0.00% GC)
  maximum time:     1.818 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     65

julia> @benchmark rand!($x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     549.856 ns (0.00% GC)
  median time:      567.626 ns (0.00% GC)
  mean time:        603.958 ns (0.00% GC)
  maximum time:     1.124 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     187

julia> @benchmark rand!(local_rng(), $x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     159.907 ns (0.00% GC)
  median time:      171.258 ns (0.00% GC)
  mean time:        174.272 ns (0.00% GC)
  maximum time:     958.197 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     788

julia> versioninfo()
Julia Version 1.6.0-DEV.1581
Commit 377aa809eb (2020-11-26 01:44 UTC)
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.0 (ORCJIT, tigerlake)
```

## Setting the seed

VectorizedRNG is initialized with a random seed (based on the default `Random.GLOBAL_RNG`) when loaded, but `Random.seed!` wont change the state of the VectorizedRNG. You can set the seed of the VectorizedRNG with `VectorizedRNG.seed!`.

```julia
julia> using VectorizedRNG

julia> rand(local_rng(), 15)'
1×15 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.580812  0.813531  0.359055  0.590277  0.551968  0.635421  0.160614  0.312387  0.00787783  0.554571  0.368705  0.0219756  0.804188  0.0740875  0.939065

julia> VectorizedRNG.seed!(1)

julia> rand(local_rng(), 15)'
1×15 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.371016  0.804553  0.243923  0.261726  0.875966  0.942672  0.875786  0.0255004  0.236359  0.59697  0.480488  0.790366  0.0263995  0.715227  0.514725

julia> rand(local_rng(), 15)'
1×15 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.246595  0.326417  0.98997  0.335991  0.839723  0.628247  0.814513  0.924231  0.398405  0.604068  0.915064  0.984332  0.773448  0.325699  0.490881

julia> VectorizedRNG.seed!(1)

julia> rand(local_rng(), 15)'
1×15 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.371016  0.804553  0.243923  0.261726  0.875966  0.942672  0.875786  0.0255004  0.236359  0.59697  0.480488  0.790366  0.0263995  0.715227  0.514725

julia> rand(local_rng(), 15)'
1×15 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.246595  0.326417  0.98997  0.335991  0.839723  0.628247  0.814513  0.924231  0.398405  0.604068  0.915064  0.984332  0.773448  0.325699  0.490881
```

## BigCrush

The generators pass [BigCrush](https://github.com/andreasnoack/RNGTest.jl). We can run BigCrush in a matter of minutes on a multicore system (10980XE CPU). Testing the uniform number generator:
```julia
julia> using Distributed; addprocs(); nprocs()
37

julia> @everywhere using RNGTest, VectorizedRNG, Random

julia> @everywhere struct U01 <: Random.AbstractRNG end

julia> @everywhere Random.rand!(r::U01, x::AbstractArray) = rand!(local_rng(), x)

julia> u01 = U01()
U01()

julia> rngunif = RNGTest.wrap(U01(), Float64);

julia> @time bcjunif = RNGTest.bigcrushJulia(rngunif);
511.531281 seconds (31.91 M allocations: 1.619 GiB, 0.10% gc time)

julia> minimum(minimum.(bcjunif))
0.004345184234132201

julia> maximum(maximum.(bcjunif))
0.99900365621945
```
and applying the cdf to the normal generator, it runs in under 10 minutes:
```julia
julia> using Distributed; addprocs(); nprocs()
37

julia> @everywhere begin;
        using Random
        using VectorizedRNG
        using RNGTest
        const INVSQRT2 = 1/sqrt(2)
        @inline function normalcdf(v)
                T = eltype(v)
                T(0.5) * ( one(T) + VectorizedRNG.SIMDPirates.verf( v * INVSQRT2 ) )
        end
        function normalcdf!(x::AbstractVector{T}) where {T}
                _W, Wshift = VectorizedRNG.VectorizationBase.pick_vector_width_shift(T)
                W = VectorizedRNG.VectorizationBase.pick_vector_width_val(T)
                N = length(x)
                ptrx = pointer(x)
                i = 0
                for _ ∈ 1:(N >>> Wshift)
                        ptrxᵢ = VectorizedRNG.VectorizationBase.gep(ptrx, i)
                        v = VectorizedRNG.SIMDPirates.vload(W, ptrxᵢ)
                        VectorizedRNG.SIMDPirates.vstore!(ptrxᵢ, normalcdf(v))
                        i += _W
                end
                if i < N
                        ptrxᵢ = VectorizedRNG.VectorizationBase.gep(ptrx, i)
                        mask = VectorizedRNG.VectorizationBase.mask(T, N & (_W - 1))
                        v = VectorizedRNG.SIMDPirates.vload(W, ptrxᵢ, mask)
                        VectorizedRNG.SIMDPirates.vstore!(ptrxᵢ, normalcdf(v), mask)
                end
                x
        end
       end

julia> @everywhere struct RN01 <: Random.AbstractRNG end

julia> @everywhere Random.rand!(r::RN01, x::AbstractArray) = normalcdf!(randn!(local_rng(), x))

julia> rngnorm = RNGTest.wrap(RN01(), Float64);

julia> @time bcj = RNGTest.bigcrushJulia(rngnorm);
592.920986 seconds (9.58 M allocations: 498.928 MiB, 0.03% gc time)

julia> minimum(minimum.(bcj))
0.0007985263854837221

julia> maximum(maximum.(bcj))
0.9990856044252019
```
Given the comparatively small diference in runtimes between the uniforn and normal -> normal quantile RNG Tests, most of the runtime is spent in RNGTest rather than evaluating the random numbers.

I don't think these are great looking minimum or maximum p-values. For comparison, the default MersenneTwister:
```julia
julia> wrappedtwister = RNGTest.wrap(MersenneTwister(), Float64);

julia> @time bcjmtwister = RNGTest.bigcrushJulia(wrappedtwister);
481.782432 seconds (9.73 M allocations: 508.753 MiB, 0.04% gc time)

julia> minimum(minimum.(bcjmtwister))
0.0015850804769910467

julia> maximum(maximum.(bcjmtwister))
0.9912021397939957
```
Interestingly, this completed faster. I should've monitored clock speeds, but can say that (subjectively) the CPU fans were louder when running this benchmark, making me wonder if this is a case where downclocking of non-AVX code decreases performance.

Watch out when mixing vectorized and non-vectorized code.

***

On vectorization: the strategy is to simply have many distinct streams, and sample from them simultaneously via SIMD operations.

