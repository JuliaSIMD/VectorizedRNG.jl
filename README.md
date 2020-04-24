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
  
```
The performance advantage is thanks primarily to a fast SIMD [Box-Muller](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform) implementation; `randn(::MersenneTwister)` uses the ziggurat algorithm, which is more efficient for scalars.
With only AVX2, the `Random` underlying uniform random number generator is faster than `VectorizedRNG`:

```julia
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
  minimum time:     4.105 μs (0.00% GC)
  median time:      4.318 μs (0.00% GC)
  mean time:        4.345 μs (0.00% GC)
  maximum time:     7.111 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     7

julia> @benchmark randn!(local_rng(), $x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     1.270 μs (0.00% GC)
  median time:      1.275 μs (0.00% GC)
  mean time:        1.277 μs (0.00% GC)
  maximum time:     2.297 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     10

julia> @benchmark rand!($x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     566.435 ns (0.00% GC)
  median time:      568.386 ns (0.00% GC)
  mean time:        569.601 ns (0.00% GC)
  maximum time:     745.505 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     184

julia> @benchmark rand!(local_rng(), $x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     201.215 ns (0.00% GC)
  median time:      201.531 ns (0.00% GC)
  mean time:        201.761 ns (0.00% GC)
  maximum time:     269.386 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     590
```

## BigCrush

The generators pass [BigCrush](https://github.com/andreasnoack/RNGTest.jl). We can run BigCrush in a matter of minutes on a multicore system (10980XE CPU). Testing the uniform number generator:
```julia
julia> using Distributed; addprocs(); nprocs()
37

julia> @everywhere using RNGTest, VectorizedRNG, Random
[ Info: Precompiling RNGTest [97cc5700-e6cb-5ca1-8fb2-7f6b45264ecd]

julia> @everywhere struct U01 <: Random.AbstractRNG end

julia> @everywhere Random.rand!(r::U01, x::AbstractArray) = rand!(local_pcg(), x)

julia> u01 = U01()
U01()

julia> rngunif = RNGTest.wrap(U01(), Float64);

julia> @time bcjunif = RNGTest.bigcrushJulia(rngunif);
515.822511 seconds (31.86 M allocations: 1.633 GiB, 0.07% gc time)

julia> minimum(minimum.(bcjunif))
0.011956745927781287

julia> maximum(maximum.(bcjunif))
0.9789973072036692
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

julia> @everywhere Random.rand!(r::RN01, x::AbstractArray) = normalcdf!(randn!(local_pcg(), x))

julia> rngnorm = RNGTest.wrap(RN01(), Float64);

julia> @time bcj = RNGTest.bigcrushJulia(rngnorm);
599.973976 seconds (9.77 M allocations: 513.287 MiB, 0.02% gc time)

julia> minimum(minimum.(bcj))
0.0007634498380764132

julia> maximum(maximum.(bcj))
0.9905810414645684
```

***

On vectorization: the strategy is to simply have many distinct streams, and sample from them simultaneously via SIMD operations. The linear congrutional element of the PCG generators each use different multipliers, so that each sequences is unique.

The `local_pcg()` returns the thread-local pcg object. Each thread, as well as each Julia process with a unique `Distributed.myid()` will use a unique multiplier, up to the currently supported limit of 1024 multipliers. After this, old multipliers will begin to be recycled.
If you have an application needing more multipliers than 1024 multipliers, please file an issue or a PR (but [beware of the [multiplier's requirements](https://en.wikipedia.org/wiki/Linear_congruential_generator#c_%E2%89%A0_0)), and we can add more.
Note that each multiplier is 64 bits, and each thread will use 4*vector width number of bits. That means an AVX2 system (with 256 bit vectors) will use 16 multipliers per thread, and an AVX512 system will use 32. Thus, 1024 multipliers is enough for up to 64 threads on an AVX2 system or 32 threads on an AVX512 system to have unique multipliers.

In addition to more multipliers, projects running on distributed systems will probably also want a way of specifying which node they are running on (will `myid()` work appropriately?); it would be great if all streams are entirely unique, so a little infrastructure may be needed to manage this.

***

The implementations were inspired by:
https://github.com/lemire/simdpcg
For more on Permuted Congrutional Generators:
http://www.pcg-random.org/
http://www.pcg-random.org/blog/


