# ###
# ### These functions are not vectorized. Therefore, they default to using
# ### the reliable GLOBAL_RNG, which has the advantage of a massive state.
# ###

# # Not sure where these should live. They aren't vectorized, but at the moment this seems like the best home for them anyway.
# """
# Samples from a gamma distribution using accept/reject.
# It requires α to be greater than 1.
# """
# function randgamma_g1(rng::AbstractRNG, α::T) where {T}
#     # @show α
#     OneThird = one(T)/T(3)
#     d = α - OneThird
#     @fastmath c = OneThird / sqrt(d)
#     @fastmath while true
#         x = randn(rng, T)
#         v = one(T) + c*x
#         v < zero(T) && continue
#         v3 = v^3
#         dv3 = d*v3 # Explicitly reference Base.log to opt out of Base.FastMath.log
#         randexp(rng, T) > T(-0.5)*x^2 - d + dv3 - d*Base.log(v3) && return dv3
#     end
# end
# function randgamma(rng::AbstractRNG, α::T) where {T}
#     α < one(T) ? exp(-randexp(rng, T)/α) * randgamma_g1(rng, α+one(T)) : randgamma_g1(rng, α)
# end
# randgamma(rng::AbstractRNG, α::T, β::T) where {T} = β * randgamma(rng, α)
# randgamma(α::Real, β::Real) = randgamma(GLOBAL_vPCG, α, β)
# randgamma(α::Real) = randgamma(GLOBAL_vPCG, α)

# randbeta(rng::AbstractRNG, α::Real, β::Real) = (a = randgamma(rng, α); a / (a + randgamma(rng, β)))
# randbeta(α::Real, β::Real) = randbeta(GLOBAL_vPCG, α, β)

# @inline randchisq(rng::AbstractRNG, ν::T) where T = T(2) * randgamma(rng, T(0.5)ν)
# @inline randchisq(ν::T) where T = T(2) * randgamma(T(0.5)ν)
# @inline randchi(rng::AbstractRNG, ν) = @fastmath sqrt(randchisq(rng, ν))
# @inline randchi(ν) = @fastmath sqrt(randchisq(ν))
# randdirichlet(α) = randdirichlet(GLOBAL_vPCG, α)
# ### NTuples and SVectors are immutable (we cannot edit them), so we create a new one.
# ### Note that they are both are stack-allocated, so creating and destroying them
# ### is fast, does not register as memory allocations, and will never trigger the garbage collector.
# ### The garbage collector cleans up heap memory, and is slow.
# @inline function randdirichlet(rng::AbstractRNG, α)
#     γ = randgamma.(Ref(rng), α)
#     @fastmath typeof(γ).mutable ? γ ./= sum(γ) : γ ./ sum(γ)
# end

