# VectorizedRNG

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/VectorizedRNG.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://chriselrod.github.io/VectorizedRNG.jl/latest)
[![Build Status](https://travis-ci.com/chriselrod/VectorizedRNG.jl.svg?branch=master)](https://travis-ci.com/chriselrod/VectorizedRNG.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/chriselrod/VectorizedRNG.jl?svg=true)](https://ci.appveyor.com/project/chriselrod/VectorizedRNG-jl)
[![Codecov](https://codecov.io/gh/chriselrod/VectorizedRNG.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chriselrod/VectorizedRNG.jl)


This library provides vectorized PCG random number generators. The larger the host computers SIMD vector width, the better they will perform. On a machine with AVX-512, they are faster than [SIMD-oriented Fast Mersenne Twister (SFMT) ](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/). Base Julia uses dSFMT,  which in a few tests appears to outperform this library on avx2 systems in generating uniformly distributed random numbers.

Testing on an old haswell machine (AVX2-only):

```julia
julia> using BenchmarkTools, Random, VectorizedRNG

julia> x = Vector{Float64}(undef, 1024);

julia> @benchmark randn!($x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     6.328 μs (0.00% GC)
  median time:      6.657 μs (0.00% GC)
  mean time:        6.738 μs (0.00% GC)
  maximum time:     50.580 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     5

julia> @benchmark randn!(local_pcg(), $x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     3.304 μs (0.00% GC)
  median time:      3.311 μs (0.00% GC)
  mean time:        3.465 μs (0.00% GC)
  maximum time:     31.240 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     8
```
The performance advantage is thanks primarily to a fast SIMD [Box-Muller](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform) implementation; `randn(::MersenneTwister)` uses the ziggurat algorithm, which is more efficient for scalars.
With only AVX2, the `Random` underlying uniform random number generator is faster than `VectorizedRNG`:

```julia
julia> @benchmark rand!($x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     807.052 ns (0.00% GC)
  median time:      809.753 ns (0.00% GC)
  mean time:        823.041 ns (0.00% GC)
  maximum time:     3.731 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     97

julia> @benchmark rand!(local_pcg(), $x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     1.428 μs (0.00% GC)
  median time:      1.528 μs (0.00% GC)
  mean time:        1.529 μs (0.00% GC)
  maximum time:     23.094 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     10
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
  minimum time:     4.019 μs (0.00% GC)
  median time:      4.241 μs (0.00% GC)
  mean time:        4.283 μs (0.00% GC)
  maximum time:     8.550 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     7

julia> @benchmark randn!(local_pcg(), $x)
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     903.415 ns (0.00% GC)
  median time:      908.732 ns (0.00% GC)
  mean time:        909.480 ns (0.00% GC)
  maximum time:     1.257 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     41

julia> @benchmark rand!($x)
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     559.565 ns (0.00% GC)
  median time:      564.726 ns (0.00% GC)
  mean time:        565.247 ns (0.00% GC)
  maximum time:     662.941 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     186

julia> @benchmark rand!(local_pcg(), $x)
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     270.262 ns (0.00% GC)
  median time:      275.839 ns (0.00% GC)
  mean time:        275.366 ns (0.00% GC)
  maximum time:     337.282 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     301
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


