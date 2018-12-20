###
### These functions are not vectorized. Therefore, they default to using
### the reliable GLOBAL_RNG, which has the advantage of a massive state.
###

randgamma(α) = randgamma(Random.GLOBAL_RNG, α)
randgamma(α, β) = randgamma(Random.GLOBAL_RNG, α, β)
randgamma(rng::AbstractRNG, α, β) = randgamma(rng, α) * β
function randgamma(rng::AbstractRNG, α::T) where T
    OneThird = one(T)/T(3)
    d = α - OneThird
    @fastmath c = OneThird / sqrt(d)
    @fastmath while true
        x = randn(rng, T)
        v = one(T) + c*x
        v < zero(T) && continue
        v3 = v^3
        dv3 = d*v3
        randexp(rng, T) > T(-0.5)*x^2 - d + dv3 - d*log(v3) && return dv3
    end
end
@inline randchisq(rng::AbstractRNG, ν::T) where T = T(2) * randgamma(rng, T(0.5)ν)
@inline randchisq(ν::T) where T = T(2) * randgamma(T(0.5)ν)
@inline randchi(rng::AbstractRNG, ν) = @fastmath sqrt(randchisq(rng, ν))
@inline randchi(ν) = @fastmath sqrt(randchisq(ν))
randdirichlet(α) = randdirichlet(Random.GLOBAL_RNG, α)
### NTuples and SVectors are immutable (we cannot edit them), so we create a new one.
### Note that they are both are stack-allocated, so creating and destroying them
### is fast, does not register as memory allocations, and will never trigger the garbage collector.
### The garbage collector cleans up heap memory, and is slow.
@inline function randdirichlet(rng::AbstractRNG, α)
    γ = randgamma.(Ref(rng), α)
    @fastmath typeof(γ).mutable ? γ ./= sum(γ) : γ ./ sum(γ)
end
