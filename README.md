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


The `PCG_RXS_M_XS{N}` generator generates `N*W` bytes at a time, where `W` is vector width (in bytes), while `PCG_XSH_RR_Core{N}` generates `N*W ÷ 2` bytes at a time. `PCG_XSH_RR` is not as fast:
```julia
julia> pcg_xsh_rr = VectorizedRNG.PCG_XSH_RR_Core{4}();

julia> @benchmark rand!($pcg_xsh_rr, $x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     489.892 ns (0.00% GC)
  median time:      493.900 ns (0.00% GC)
  mean time:        513.294 ns (0.00% GC)
  maximum time:     748.369 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     195

julia> @benchmark rand!($pcg_xsh_rr, $x32)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     250.728 ns (0.00% GC)
  median time:      251.989 ns (0.00% GC)
  mean time:        263.913 ns (0.00% GC)
  maximum time:     398.616 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     372
```
but it is cryptographically secure, because it discards half of the bytes. This also gives it a longer period.

They can also be used to generate `NTuple{N,Core.VecElement{T}}` (SIMD) vectors directly, without needing to store them on the heap:

```julia
julia> @benchmark rand($pcg_rxs_m_xs, NTuple{32,Core.VecElement{Float64}})
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  4544017950713774080
  --------------
  minimum time:     8.894 ns (0.00% GC)
  median time:      8.951 ns (0.00% GC)
  mean time:        9.355 ns (0.00% GC)
  maximum time:     34.965 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     999

julia> @benchmark rand($pcg_rxs_m_xs, NTuple{64,Core.VecElement{Float32}})
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  4082301957031066420
  --------------
  minimum time:     9.090 ns (0.00% GC)
  median time:      9.623 ns (0.00% GC)
  mean time:        9.910 ns (0.00% GC)
  maximum time:     33.498 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     999

julia> @benchmark rand($pcg_xsh_rr, NTuple{16,Core.VecElement{Float64}})
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  4554829761946574848
  --------------
  minimum time:     7.331 ns (0.00% GC)
  median time:      7.376 ns (0.00% GC)
  mean time:        7.673 ns (0.00% GC)
  maximum time:     40.476 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     999

julia> @benchmark rand($pcg_xsh_rr, NTuple{32,Core.VecElement{Float32}})
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  4071394801663940768
  --------------
  minimum time:     7.322 ns (0.00% GC)
  median time:      7.376 ns (0.00% GC)
  mean time:        7.727 ns (0.00% GC)
  maximum time:     30.680 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     999

julia> randv(::Type{NTuple{N,Core.VecElement{T}}}) where {N,T} = ntuple(Val(N)) do x Core.VecElement{T}(rand(T)) end
randv (generic function with 1 method)

julia> @benchmark randv(NTuple{16,Core.VecElement{Float64}})
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  4544176676982603776
  --------------
  minimum time:     19.633 ns (0.00% GC)
  median time:      20.227 ns (0.00% GC)
  mean time:        20.872 ns (0.00% GC)
  maximum time:     54.304 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     999

julia> @benchmark randv(NTuple{32,Core.VecElement{Float32}})
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  4045780578808817602
  --------------
  minimum time:     40.825 ns (0.00% GC)
  median time:      41.462 ns (0.00% GC)
  mean time:        42.816 ns (0.00% GC)
  maximum time:     90.207 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     996
```
They can be used to generate random numbers in multiples of `W` bytes at a time.
If generating large numbers of random numbers in a (partially unrolled loop), this can be useful $-$ and save you from having to cache random numbers in a vector and load them. It will (as shown here) perform much better than the base implementation.

They are however fastest when generating the numbers in chunks of `N*W` and `N*W ÷ 2` at a time, thanks to the CPU taking advantage of out of order execution. This probably plays well with mixing floating point operations in between (eg, within the for loop), but that will take more testing.

These times are much better than Julia's base implementation gives us for small vectors:
```julia
julia> x64_32 = Vector{Float64}(undef, 32);

julia> @benchmark rand!($x64_32)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     38.780 ns (0.00% GC)
  median time:      39.406 ns (0.00% GC)
  mean time:        40.678 ns (0.00% GC)
  maximum time:     77.330 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     996
```
Because dSFMT needs larger vectors to be fast.

To fill out arrays whose length isn't a multiple of the vector width, the rng simply generates extra numbers and discards the rest.
```julia
julia> x97 = Vector{Float64}(undef, 97);

julia> @benchmark rand!($pcg_rxs_m_xs, $x97)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     49.461 ns (0.00% GC)
  median time:      49.650 ns (0.00% GC)
  mean time:        51.749 ns (0.00% GC)
  maximum time:     89.897 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     987

julia> @benchmark rand!($pcg_xsh_rr, $x97)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     65.456 ns (0.00% GC)
  median time:      65.910 ns (0.00% GC)
  mean time:        68.823 ns (0.00% GC)
  maximum time:     124.193 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     979

julia> @benchmark rand!($x97) # dSFMT
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     80.823 ns (0.00% GC)
  median time:      81.651 ns (0.00% GC)
  mean time:        85.348 ns (0.00% GC)
  maximum time:     130.992 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     976

```
but the number discarded isn't too severe.

The generators also pass BigCrush:
```julia
julia> using RNGTest

julia> pcg_rxs_m_xs_wrapped = RNGTest.wrap(pcg_rxs_m_xs, Float64);

julia> bcres = RNGTest.bigcrushJulia(pcg_rxs_m_xs_wrapped);

julia> for BigCrushResult ∈ bcres
           @show BigCrushResult
       end
BigCrushResult = 0.4106591309481814
BigCrushResult = 0.17354345054361753
BigCrushResult = 0.25482276442280055
BigCrushResult = 0.7730928805307954
BigCrushResult = 0.4203880554047008
BigCrushResult = 0.338376931674389
BigCrushResult = 0.8822040862633121
BigCrushResult = 0.5061837335048374
BigCrushResult = 0.9818260873845434
BigCrushResult = 0.15019183540128808
BigCrushResult = 0.9632585526038174
BigCrushResult = 0.02438292010757134
BigCrushResult = 0.029463347657525285
BigCrushResult = 0.3363028569795805
BigCrushResult = 0.15422145446376959
BigCrushResult = 0.2778687870888937
BigCrushResult = 0.7241694838710344
BigCrushResult = 0.13822136411214775
BigCrushResult = 0.8832839811695422
BigCrushResult = 0.41078309316645983
BigCrushResult = 0.7502726060531351
BigCrushResult = (0.808051199336255, 0.7985607391748085, 0.662235457387914, 0.6475022923276412)
BigCrushResult = (0.1723541531903923, 0.23429209565636291, 0.037817712530549726, 0.7614923705297416)
BigCrushResult = (0.7766083071693258, 0.5807899978031251, 0.16729025105112105, 0.507677788886264)
BigCrushResult = (0.8676418263825746, 0.9693213070893868, 0.3519310160640892, 0.5108582297493598)
BigCrushResult = 0.2770474865852386
BigCrushResult = 0.9368888939677381
BigCrushResult = 0.27321308848093495
BigCrushResult = 0.9822040981229053
BigCrushResult = 0.037989439995602114
BigCrushResult = 0.9764232905796569
BigCrushResult = 0.6644615555151838
BigCrushResult = 0.7360731854488078
BigCrushResult = 0.9839178667185824
BigCrushResult = 0.6846680444938233
BigCrushResult = 0.33102760813241894
BigCrushResult = 0.13405701074521836
BigCrushResult = 0.36944145461953526
BigCrushResult = 0.2960943744492237
BigCrushResult = 0.1975964378414432
BigCrushResult = 0.06690127141962932
BigCrushResult = 0.06417205758717137
BigCrushResult = 0.9547303269337241
BigCrushResult = 0.9343303350351551
BigCrushResult = 0.6625420979693456
BigCrushResult = (0.11654949135288162, 0.09080578027093747)
BigCrushResult = (0.38342755040501036, 0.11619194130020431)
BigCrushResult = (0.33580273511618286, 0.3988350835817301)
BigCrushResult = (0.08105517583498037, 0.3417448068186771)
BigCrushResult = 0.8932419654456486
BigCrushResult = 0.6714096067094054
BigCrushResult = 0.6008737137471256
BigCrushResult = 0.4664362075970923
BigCrushResult = 0.6857383826965994
BigCrushResult = 0.1081544982178867
BigCrushResult = 0.9445970430262466
BigCrushResult = 0.384508509704899
BigCrushResult = 0.07147833791901248
BigCrushResult = 0.24447806788204274
BigCrushResult = 0.23128427883686098
BigCrushResult = 0.9397200851801257
BigCrushResult = 0.48205491887721197
BigCrushResult = 0.8313844233060546
BigCrushResult = 0.891905748635546
BigCrushResult = 0.1905642757196756
BigCrushResult = 0.9854356329153938
BigCrushResult = 0.7730422115767813
BigCrushResult = 0.7636170894533159
BigCrushResult = 0.5373742165051967
BigCrushResult = 0.9710549453220593
BigCrushResult = 0.4805890997309501
BigCrushResult = 0.6938445416901984
BigCrushResult = 0.0382344488380024
BigCrushResult = (0.04600478369223504, 0.3560079152587359, 0.5920867567186807, 0.42905097366650113, 0.9913329182009624)
BigCrushResult = (0.7648747227881576, 0.4756202989842768, 0.7163165830201237, 0.6315588032916277, 0.6499679746583439)
BigCrushResult = (0.5235911753090674, 0.00029860512604518163, 0.045821291143107734, 0.39385168287231664, 0.9042729541850227)
BigCrushResult = (0.8235009653149202, 0.7690716068317154, 0.5877922055394105, 0.5730408188573479, 0.6850443711992608)
BigCrushResult = (0.19714994293932808, 0.434345240732412, 0.8134951872465254, 0.46846150982174994, 0.38014048996429683)
BigCrushResult = (0.42230277041451414, 0.8541574590031198, 0.23333661167442177, 0.32120812318176073, 0.8887677770904957)
BigCrushResult = (0.08034374768174612, 0.09586344522813983)
BigCrushResult = (0.6891444622081704, 0.7374127230443792)
BigCrushResult = 0.06683421687976276
BigCrushResult = 0.08459536060915107
BigCrushResult = 0.7564698344007108
BigCrushResult = 0.7108533285240521
BigCrushResult = (0.8500367295323077, 0.6878130593504515)
BigCrushResult = (0.6728923566216871, 0.6878130593504515)
BigCrushResult = 0.3426091501560981
BigCrushResult = 0.9550998747052178
BigCrushResult = 0.9891990125379503
BigCrushResult = 0.19073227961424313
BigCrushResult = 0.05261013172312623
BigCrushResult = 0.04435806608777859
BigCrushResult = 0.6757043317391018
BigCrushResult = 0.6885848634853019
BigCrushResult = 0.4916365012613815
BigCrushResult = 0.5583068144235634
BigCrushResult = 0.5614789645785124
BigCrushResult = 0.1800460594691734
BigCrushResult = 0.25752830837792773
BigCrushResult = (0.564979588995816, 0.8446614991055403)
BigCrushResult = (0.10466546158466428, 0.43282680090448267)
BigCrushResult = 0.1795713199409282
BigCrushResult = 0.23008710998634221
BigCrushResult = 0.20320720460304584
BigCrushResult = 0.029796016529786768
```
The `pcg_xsh_rr` generator passes as well.

Finally, `randexp` and `randn` are implemented. While `Random` uses the ziggurat algorithm, I've simply implemented with via the probability integral transform and box-muller algorithm, using [SLEEF](https://sleef.org/) through [SLEEFwrap.jl](https://github.com/chriselrod/SLEEFwrap.jl) for the vectorized elementary functions.

```julia
julia> using VectorizedRNG, Random, BenchmarkTools

julia> @benchmark randexp($pcg_rxs_m_xs, NTuple{32,Core.VecElement{Float64}})
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  4552730547917148817
  --------------
  minimum time:     35.862 ns (0.00% GC)
  median time:      36.052 ns (0.00% GC)
  mean time:        37.374 ns (0.00% GC)
  maximum time:     72.542 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     993

julia> @benchmark randn($pcg_rxs_m_xs, NTuple{32,Core.VecElement{Float64}})
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  -4665825264741215482
  --------------
  minimum time:     55.082 ns (0.00% GC)
  median time:      55.765 ns (0.00% GC)
  mean time:        57.708 ns (0.00% GC)
  maximum time:     95.163 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     984

julia> @benchmark randexp($pcg_rxs_m_xs, NTuple{64,Core.VecElement{Float32}})
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  4041981444005099904
  --------------
  minimum time:     28.145 ns (0.00% GC)
  median time:      28.362 ns (0.00% GC)
  mean time:        29.609 ns (0.00% GC)
  maximum time:     63.881 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     993

julia> @benchmark randn($pcg_rxs_m_xs, NTuple{64,Core.VecElement{Float32}})
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  -5279861454082162424
  --------------
  minimum time:     41.599 ns (0.00% GC)
  median time:      41.705 ns (0.00% GC)
  mean time:        43.221 ns (0.00% GC)
  maximum time:     79.216 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     991

julia> @benchmark randexp($pcg_xsh_rr, NTuple{16,Core.VecElement{Float64}})
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  4553980226834452730
  --------------
  minimum time:     23.384 ns (0.00% GC)
  median time:      23.484 ns (0.00% GC)
  mean time:        24.398 ns (0.00% GC)
  maximum time:     57.557 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     996

julia> @benchmark randn($pcg_xsh_rr, NTuple{16,Core.VecElement{Float64}})
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  -4674448221349828633
  --------------
  minimum time:     36.729 ns (0.00% GC)
  median time:      36.986 ns (0.00% GC)
  mean time:        38.401 ns (0.00% GC)
  maximum time:     85.148 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     992

julia> @benchmark randexp($pcg_xsh_rr, NTuple{32,Core.VecElement{Float32}})
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  4064078234704277841
  --------------
  minimum time:     19.692 ns (0.00% GC)
  median time:      19.788 ns (0.00% GC)
  mean time:        20.529 ns (0.00% GC)
  maximum time:     53.294 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     997

julia> @benchmark randn($pcg_xsh_rr, NTuple{32,Core.VecElement{Float32}})
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  -5190744380927662965
  --------------
  minimum time:     28.649 ns (0.00% GC)
  median time:      30.591 ns (0.00% GC)
  mean time:        31.487 ns (0.00% GC)
  maximum time:     70.541 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     995
```

These are also much faster for filling vectors than `Random`'s methods:
```julia
julia> x = Vector{Float64}(undef, 1024);

julia> x32 = Vector{Float32}(undef, 1024);

julia> @benchmark randn!($pcg_rxs_m_xs, $x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     1.777 μs (0.00% GC)
  median time:      1.795 μs (0.00% GC)
  mean time:        1.863 μs (0.00% GC)
  maximum time:     4.572 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     10

julia> @benchmark randn!($pcg_xsh_rr, $x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     2.336 μs (0.00% GC)
  median time:      2.355 μs (0.00% GC)
  mean time:        2.449 μs (0.00% GC)
  maximum time:     6.030 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     9

julia> @benchmark randn!($x)  # dSFMT + Random's Ziggurat
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     4.010 μs (0.00% GC)
  median time:      4.230 μs (0.00% GC)
  mean time:        4.369 μs (0.00% GC)
  maximum time:     9.258 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     7

julia> @benchmark randn!($pcg_rxs_m_xs, $x32)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     685.900 ns (0.00% GC)
  median time:      699.140 ns (0.00% GC)
  mean time:        739.964 ns (0.00% GC)
  maximum time:     1.093 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     150

julia> @benchmark randn!($pcg_xsh_rr, $x32)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     892.622 ns (0.00% GC)
  median time:      896.933 ns (0.00% GC)
  mean time:        930.740 ns (0.00% GC)
  maximum time:     1.673 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     45

julia> @benchmark randn!($x32)  # dSFMT + Random's Ziggurat
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     4.083 μs (0.00% GC)
  median time:      4.296 μs (0.00% GC)
  mean time:        4.420 μs (0.00% GC)
  maximum time:     9.934 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     7

julia> @benchmark randexp!($pcg_rxs_m_xs, $x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     1.185 μs (0.00% GC)
  median time:      1.191 μs (0.00% GC)
  mean time:        1.237 μs (0.00% GC)
  maximum time:     4.544 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     10

julia> @benchmark randexp!($pcg_xsh_rr, $x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     1.476 μs (0.00% GC)
  median time:      1.485 μs (0.00% GC)
  mean time:        1.563 μs (0.00% GC)
  maximum time:     4.887 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     10

julia> @benchmark randexp!($x)  # dSFMT + Random's Ziggurat
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     3.876 μs (0.00% GC)
  median time:      4.127 μs (0.00% GC)
  mean time:        4.220 μs (0.00% GC)
  maximum time:     9.209 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     7

julia> @benchmark randexp!($pcg_rxs_m_xs, $x32)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     473.582 ns (0.00% GC)
  median time:      474.556 ns (0.00% GC)
  mean time:        493.378 ns (0.00% GC)
  maximum time:     718.821 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     196

julia> @benchmark randexp!($pcg_xsh_rr, $x32)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     632.805 ns (0.00% GC)
  median time:      633.941 ns (0.00% GC)
  mean time:        658.469 ns (0.00% GC)
  maximum time:     1.018 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     169

julia> @benchmark randexp!($x32)  # dSFMT + Random's Ziggurat
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     3.985 μs (0.00% GC)
  median time:      4.312 μs (0.00% GC)
  mean time:        4.467 μs (0.00% GC)
  maximum time:     9.430 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     7

```

***

On vectorization: the strategy is to simply have a bunch of distinct streams, and sample from them simultaneously via SIMD operations. The linear congrutional element of the PCG generators each use different increments, so that the sequences are somewhat different.

Ideally, each one would use different multipliers too $-$ that would give us truly independent streams. The rule for increments is just that they have to be odd, but multipliers are more complicated.
See [here](http://www.pcg-random.org/posts/critiquing-pcg-streams.html) for more information.
I'm considering picking a few thousand random integers, and running `RNGTest.sspectral_Fourier3` on the LCGs, then have the PCGs go down this list on initialization.

Similarly $-$ especially if we get different multiplicative constants too $-$ they're a natural choice for multithreaded or distributed operations. However, for now you'll have to specify the increments (and ideally different multiplicative constants) yourself.

***

The implementations were inspired by:
https://github.com/lemire/simdpcg
For more on Permuted Congrutional Generators:
http://www.pcg-random.org/
http://www.pcg-random.org/blog/
