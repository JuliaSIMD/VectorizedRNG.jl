
const XREGISTERS = 4
const XSTREAMS = W64 * XREGISTERS

struct Xoshift{P} <: AbstractVRNG{P}
    ptr::Ptr{UInt64}
end
@inline Base.pointer(rng::Xoshift) = rng.ptr

struct XoshiftState{P,W} <: AbstractState{P,W}
    eins::NTuple{P,SVec{W,UInt64}}
    zwei::NTuple{P,SVec{W,UInt64}}
    drei::NTuple{P,SVec{W,UInt64}}
    vier::NTuple{P,SVec{W,UInt64}}
end

Xoshift(ptr) = Xoshift{XSTREAMS}(ptr)
function randnonzero()
    while true
        r = rand(UInt64)
        iszero(r) || return r
    end
end
function initXoshift!(ptr::Ptr{UInt64}, P)
    e = randnonzero(); z = randnonzero();
    d = randnonzero(); v = randnonzero();
    initXoshift!(ptr, P, e, z, d, v)
end
function initXoshift!(ptr::Ptr{UInt64}, P, e::UInt64, z::UInt64, d::UInt64, v::UInt64) # P here means number of streams
    for j ∈ 1:P-1
        i = P - j
        vstore!(ptr, e, 8i); vstore!(ptr, z, 8*(i + P)); vstore!(ptr, d, 8*(i + 2P)); vstore!(ptr, v, 8*(i + 3P));
        e, z, d, v = jump(e, z, d, v)
    end
    vstore!(ptr, e); vstore!(ptr, z, 8P); vstore!(ptr, d, 8*(2P)); vstore!(ptr, v, 8*(3P));
end
function jump(eins, zwei, drei, vier)
    e = zero(UInt64); z = zero(UInt64); d = zero(UInt64); v = zero(UInt64)
    for u ∈ (0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c)
        for b ∈ 0:63
            if u % Bool
                e ⊻= eins
                z ⊻= zwei
                d ⊻= drei
                v ⊻= vier
            end
            u >>>= 1
            e, z, d, v = nextstate(e, z, d, v)
        end
    end
    e, z, d, v
end
function seed!(seed::Integer) 
    i = seed % UInt64
    e = z = d = v = zero(UInt64)
    increment = 0xa04de531e612e1b9
    while any(iszero, (e, z, d, v))
        e = ((i * 0x90ce6ecbad5e33b5) + increment)
        z = ((e * 0x90ce6ecbad5e33b5) + increment)
        d = ((z * 0x90ce6ecbad5e33b5) + increment)
        v = ((d * 0x90ce6ecbad5e33b5) + increment)
        increment += 0x0000000000000002
    end
    nstreams = XREGISTERS * Base.Threads.nthreads() * W64
    initXoshift!(GLOBAL_vRNGs[], nstreams, e, z, d, v)
end


# @inline function getstate(rng::Xoshift{P}, ::Val{N}, ::Val{W}) where {P,N,W}
#     ptr = pointer(rng)
#     XoshiftState(
#         ntuple(n -> vloada(SVec{W,UInt64}, ptr, W64*(n - 1)), Val{N}()),
#         ntuple(n -> vloada(SVec{W,UInt64}, ptr, W64*(n - 1) + W64*P), Val{N}()),
#         ntuple(n -> vloada(SVec{W,UInt64}, ptr, W64*(n - 1) + 2W64*P), Val{N}()),
#         ntuple(n -> vloada(SVec{W,UInt64}, ptr, W64*(n - 1) + 3W64*P), Val{N}())
#     )
# end
@inline function getstate(rng::Xoshift{P}, ::Val{1}, ::Val{W}) where {P,W}
    ptr = pointer(rng)
    XoshiftState(
        (vloada(SVec{W,UInt64}, ptr),),
        (vloada(SVec{W,UInt64}, ptr,  REGISTER_SIZE*P),),
        (vloada(SVec{W,UInt64}, ptr, 2REGISTER_SIZE*P),),
        (vloada(SVec{W,UInt64}, ptr, 3REGISTER_SIZE*P),)
    )
end
@inline function getstate(rng::Xoshift{P}, ::Val{2}, ::Val{W}) where {P,W}
    ptr = pointer(rng)
    XoshiftState(
        (vloada(SVec{W,UInt64}, ptr                  ), vloada(SVec{W,UInt64}, ptr, REGISTER_SIZE         )),
        (vloada(SVec{W,UInt64}, ptr,  P*REGISTER_SIZE), vloada(SVec{W,UInt64}, ptr, REGISTER_SIZE*(1 +  P))),
        (vloada(SVec{W,UInt64}, ptr, 2P*REGISTER_SIZE), vloada(SVec{W,UInt64}, ptr, REGISTER_SIZE*(1 + 2P))),
        (vloada(SVec{W,UInt64}, ptr, 3P*REGISTER_SIZE), vloada(SVec{W,UInt64}, ptr, REGISTER_SIZE*(1 + 3P)))
    )
end
@inline function getstate(rng::Xoshift{P}, ::Val{4}, ::Val{W}) where {P,W}
    ptr = pointer(rng)
    RS = REGISTER_SIZE
    XoshiftState(
        (vloada(SVec{W,UInt64}, ptr        ), vloada(SVec{W,UInt64}, ptr, RS         ), vloada(SVec{W,UInt64}, ptr, RS* 2      ), vloada(SVec{W,UInt64}, ptr, 3RS        )),
        (vloada(SVec{W,UInt64}, ptr,  P*RS), vloada(SVec{W,UInt64}, ptr, RS*(1 +  P)), vloada(SVec{W,UInt64}, ptr, RS*(2 +  P)), vloada(SVec{W,UInt64}, ptr, RS*(3 +  P))),
        (vloada(SVec{W,UInt64}, ptr, 2P*RS), vloada(SVec{W,UInt64}, ptr, RS*(1 + 2P)), vloada(SVec{W,UInt64}, ptr, RS*(2 + 2P)), vloada(SVec{W,UInt64}, ptr, RS*(3 + 2P))),
        (vloada(SVec{W,UInt64}, ptr, 3P*RS), vloada(SVec{W,UInt64}, ptr, RS*(1 + 3P)), vloada(SVec{W,UInt64}, ptr, RS*(2 + 3P)), vloada(SVec{W,UInt64}, ptr, RS*(3 + 3P)))
    )
end
@inline function storestate!(rng::Xoshift{P}, s::XoshiftState{N,W}) where {P,N,W}
    ptr = pointer(rng)
    @unpack eins, zwei, drei, vier = s
    @inbounds for n ∈ 1:N
        vstorea!(rng, eins[n], REGISTER_SIZE*(n-1))
    end
    @inbounds for n ∈ 1:N
        vstorea!(rng, zwei[n], REGISTER_SIZE*((n-1) +  P))
    end
    @inbounds for n ∈ 1:N
        vstorea!(rng, drei[n], REGISTER_SIZE*((n-1) + 2P))
    end
    @inbounds for n ∈ 1:N
        vstorea!(rng, vier[n], REGISTER_SIZE*((n-1) + 3P))
    end
end
@inline function storestate!(rng::Xoshift{P}, s::XoshiftState{1,W}) where {P,W}
    ptr = pointer(rng)
    @unpack eins, zwei, drei, vier = s;
    @inbounds begin
        vstorea!(ptr, eins[1],       )
        vstorea!(ptr, zwei[1],  P*REGISTER_SIZE)
        vstorea!(ptr, drei[1], 2P*REGISTER_SIZE)
        vstorea!(ptr, vier[1], 3P*REGISTER_SIZE)
    end
end
@inline function storestate!(rng::Xoshift{P}, s::XoshiftState{2,W}) where {P,W}
    ptr = pointer(rng)
    @unpack eins, zwei, drei, vier = s;
    @inbounds begin
        vstorea!(ptr, eins[1],      )
        vstorea!(ptr, eins[2],   REGISTER_SIZE)
        vstorea!(ptr, zwei[1],   REGISTER_SIZE*       P)
        vstorea!(ptr, zwei[2],   REGISTER_SIZE*(1 +   P))
        vstorea!(ptr, drei[1],   REGISTER_SIZE*      2P)
        vstorea!(ptr, drei[2],   REGISTER_SIZE*(1 +  2P))
        vstorea!(ptr, vier[1],   REGISTER_SIZE*      3P)
        vstorea!(ptr, vier[2],   REGISTER_SIZE*(1 +  3P))
    end
end
@inline function storestate!(rng::Xoshift{P}, s::XoshiftState{4,W}) where {P,W}
    ptr = pointer(rng)
    @unpack eins, zwei, drei, vier = s;
    @inbounds begin
        vstorea!(ptr, eins[1],      )
        vstorea!(ptr, eins[2],   REGISTER_SIZE)
        vstorea!(ptr, eins[3],   REGISTER_SIZE*2)
        vstorea!(ptr, eins[4],   REGISTER_SIZE*3)
        vstorea!(ptr, zwei[1],   REGISTER_SIZE*       P)
        vstorea!(ptr, zwei[2],   REGISTER_SIZE*(1 +   P))
        vstorea!(ptr, zwei[3],   REGISTER_SIZE*(2 +   P))
        vstorea!(ptr, zwei[4],   REGISTER_SIZE*(3 +   P))
        vstorea!(ptr, drei[1],   REGISTER_SIZE*      2P)
        vstorea!(ptr, drei[2],   REGISTER_SIZE*(1 +  2P))
        vstorea!(ptr, drei[3],   REGISTER_SIZE*(2 +  2P))
        vstorea!(ptr, drei[4],   REGISTER_SIZE*(3 +  2P))
        vstorea!(ptr, vier[1],   REGISTER_SIZE*      3P)
        vstorea!(ptr, vier[2],   REGISTER_SIZE*(1 +  3P))
        vstorea!(ptr, vier[3],   REGISTER_SIZE*(2 +  3P))
        vstorea!(ptr, vier[4],   REGISTER_SIZE*(3 +  3P))
    end
end

@inline function nextstate(eins, zwei, drei, vier)
    t = zwei << 0x0000000000000011
    drei = drei ⊻ eins
    vier = vier ⊻ zwei
    zwei = zwei ⊻ drei
    eins = eins ⊻ vier
    drei = drei ⊻ t
    vier = rotate_right(vier, 0x000000000000002d)
    eins, zwei, drei, vier
end

@inline function nextstate(s::XoshiftState{P}, ::Val{1}) where {P}
    (eins,) = s.eins;
    (zwei,) = s.zwei;
    (drei,) = s.drei;
    (vier,) = s.vier;
    # out = vmul(zwei, 0x000000000000005)
    # out = rotate_right(out, 0x0000000000000007)
    # out = vmul(out, 0x0000000000000009)
    out = vadd(eins, vier)
    out = rotate_right(out, 0x0000000000000017)
    eins, zwei, drei, vier = nextstate(eins, zwei, drei, vier)
    out = vadd(eins, out)
    XoshiftState( (eins,), (zwei,), (drei,), (vier,)), (out,)
end

@inline function nextstate(s::XoshiftState{2}, ::Val{2})
    (Base.Cartesian.@ntuple 2 eins) = s.eins;
    (Base.Cartesian.@ntuple 2 zwei) = s.zwei;
    (Base.Cartesian.@ntuple 2 drei) = s.drei;
    (Base.Cartesian.@ntuple 2 vier) = s.vier;
    # Base.Cartesian.@nexprs 2 n -> out_n = vmul(zwei_n, 0x000000000000005)
    # Base.Cartesian.@nexprs 2 n -> out_n = rotate_right(out_n, 0x0000000000000007)
    # Base.Cartesian.@nexprs 2 n -> out_n = vmul(out_n, 0x0000000000000009)
    Base.Cartesian.@nexprs 2 n -> out_n = vadd(eins_n, vier_n)
    Base.Cartesian.@nexprs 2 n -> t_n = zwei_n << 0x0000000000000011
    Base.Cartesian.@nexprs 2 n -> out_n = rotate_right(out_n, 0x0000000000000017)
    Base.Cartesian.@nexprs 2 n -> drei_n = drei_n ⊻ eins_n
    Base.Cartesian.@nexprs 2 n -> vier_n = vier_n ⊻ zwei_n
    Base.Cartesian.@nexprs 2 n -> out_n = vadd(eins_n, out_n)
    Base.Cartesian.@nexprs 2 n -> zwei_n = zwei_n ⊻ drei_n
    Base.Cartesian.@nexprs 2 n -> eins_n = eins_n ⊻ vier_n
    Base.Cartesian.@nexprs 2 n -> drei_n = drei_n ⊻ t_n
    Base.Cartesian.@nexprs 2 n -> vier_n = rotate_right(vier_n, 0x000000000000002d)
    XoshiftState(
        (Base.Cartesian.@ntuple 2 eins),
        (Base.Cartesian.@ntuple 2 zwei),
        (Base.Cartesian.@ntuple 2 drei),
        (Base.Cartesian.@ntuple 2 vier)
    ), Base.Cartesian.@ntuple 2 out
end
@inline function nextstate(s::XoshiftState{4}, ::Val{2})
    (Base.Cartesian.@ntuple 4 eins) = s.eins;
    (Base.Cartesian.@ntuple 4 zwei) = s.zwei;
    (Base.Cartesian.@ntuple 4 drei) = s.drei;
    (Base.Cartesian.@ntuple 4 vier) = s.vier;
    # Base.Cartesian.@nexprs 2 n -> out_n = vmul(zwei_n, 0x000000000000005)
    # Base.Cartesian.@nexprs 2 n -> out_n = rotate_right(out_n, 0x0000000000000007)
    # Base.Cartesian.@nexprs 2 n -> out_n = vmul(out_n, 0x0000000000000009)
    Base.Cartesian.@nexprs 2 n -> out_n = vadd(eins_n, vier_n)
    Base.Cartesian.@nexprs 2 n -> out_n = rotate_right(out_n, 0x0000000000000017)
    Base.Cartesian.@nexprs 2 n -> out_n = vadd(eins_n, out_n)
    Base.Cartesian.@nexprs 2 n -> t_n = zwei_n << 0x0000000000000011
    Base.Cartesian.@nexprs 2 n -> drei_n = drei_n ⊻ eins_n
    Base.Cartesian.@nexprs 2 n -> vier_n = vier_n ⊻ zwei_n
    Base.Cartesian.@nexprs 2 n -> zwei_n = zwei_n ⊻ drei_n
    Base.Cartesian.@nexprs 2 n -> eins_n = eins_n ⊻ vier_n
    Base.Cartesian.@nexprs 2 n -> drei_n = drei_n ⊻ t_n
    Base.Cartesian.@nexprs 2 n -> vier_n = rotate_right(vier_n, 0x000000000000002d)
    XoshiftState(
        (Base.Cartesian.@ntuple 4 eins),
        (Base.Cartesian.@ntuple 4 zwei),
        (Base.Cartesian.@ntuple 4 drei),
        (Base.Cartesian.@ntuple 4 vier)
    ), Base.Cartesian.@ntuple 2 out
end
@inline function nextstate(s::XoshiftState{4}, ::Val{3})
    (Base.Cartesian.@ntuple 4 eins) = s.eins;
    (Base.Cartesian.@ntuple 4 zwei) = s.zwei;
    (Base.Cartesian.@ntuple 4 drei) = s.drei;
    (Base.Cartesian.@ntuple 4 vier) = s.vier;
    # Base.Cartesian.@nexprs 3 n -> out_n = vmul(zwei_n, 0x000000000000005)
    # Base.Cartesian.@nexprs 3 n -> out_n = rotate_right(out_n, 0x0000000000000007)
    # Base.Cartesian.@nexprs 3 n -> out_n = vmul(out_n, 0x0000000000000009)
    Base.Cartesian.@nexprs 3 n -> out_n = vadd(eins_n, vier_n)
    Base.Cartesian.@nexprs 3 n -> out_n = rotate_right(out_n, 0x0000000000000017)
    Base.Cartesian.@nexprs 3 n -> out_n = vadd(eins_n, out_n)
    Base.Cartesian.@nexprs 3 n -> t_n = zwei_n << 0x0000000000000011
    Base.Cartesian.@nexprs 3 n -> drei_n = drei_n ⊻ eins_n
    Base.Cartesian.@nexprs 3 n -> vier_n = vier_n ⊻ zwei_n
    Base.Cartesian.@nexprs 3 n -> zwei_n = zwei_n ⊻ drei_n
    Base.Cartesian.@nexprs 3 n -> eins_n = eins_n ⊻ vier_n
    Base.Cartesian.@nexprs 3 n -> drei_n = drei_n ⊻ t_n
    Base.Cartesian.@nexprs 3 n -> vier_n = rotate_right(vier_n, 0x000000000000002d)
    XoshiftState(
        (Base.Cartesian.@ntuple 4 eins),
        (Base.Cartesian.@ntuple 4 zwei),
        (Base.Cartesian.@ntuple 4 drei),
        (Base.Cartesian.@ntuple 4 vier)
    ), Base.Cartesian.@ntuple 3 out
end
@inline function nextstate(s::XoshiftState{4}, ::Val{4})
    (Base.Cartesian.@ntuple 4 eins) = s.eins;
    (Base.Cartesian.@ntuple 4 zwei) = s.zwei;
    (Base.Cartesian.@ntuple 4 drei) = s.drei;
    (Base.Cartesian.@ntuple 4 vier) = s.vier;
    # Base.Cartesian.@nexprs 4 n -> out_n = vmul(zwei_n, 0x000000000000005)
    # Base.Cartesian.@nexprs 4 n -> out_n = rotate_right(out_n, 0x0000000000000007)
    # Base.Cartesian.@nexprs 4 n -> out_n = vmul(out_n, 0x0000000000000009)
    Base.Cartesian.@nexprs 4 n -> out_n = vadd(eins_n, vier_n)
    Base.Cartesian.@nexprs 4 n -> out_n = rotate_right(out_n, 0x0000000000000017)
    Base.Cartesian.@nexprs 4 n -> out_n = vadd(eins_n, out_n)
    Base.Cartesian.@nexprs 4 n -> t_n = zwei_n << 0x0000000000000011
    Base.Cartesian.@nexprs 4 n -> drei_n = drei_n ⊻ eins_n
    Base.Cartesian.@nexprs 4 n -> vier_n = vier_n ⊻ zwei_n
    Base.Cartesian.@nexprs 4 n -> zwei_n = zwei_n ⊻ drei_n
    Base.Cartesian.@nexprs 4 n -> eins_n = eins_n ⊻ vier_n
    Base.Cartesian.@nexprs 4 n -> drei_n = drei_n ⊻ t_n
    Base.Cartesian.@nexprs 4 n -> vier_n = rotate_right(vier_n, 0x000000000000002d)
    XoshiftState(
        (Base.Cartesian.@ntuple 4 eins),
        (Base.Cartesian.@ntuple 4 zwei),
        (Base.Cartesian.@ntuple 4 drei),
        (Base.Cartesian.@ntuple 4 vier)
    ), Base.Cartesian.@ntuple 4 out
end
@inline function nextstate(s::XoshiftState{2}, ::Val{3})
    s, (out_1,out_2) = nextstate(s, Val(2))
    s, (out_3,) = nextstate(s, Val(1))
    s, Base.Cartesian.@ntuple 3 out
end
@inline function nextstate(s::XoshiftState{2}, ::Val{4})
    s, (out_1,out_2) = nextstate(s, Val(2))
    s, (out_3,out_4) = nextstate(s, Val(2))
    s, Base.Cartesian.@ntuple 4 out
end

