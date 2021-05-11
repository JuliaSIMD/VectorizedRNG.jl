
const XREGISTERS = 2

struct Xoshift{P} <: AbstractVRNG{P}
    ptr::Ptr{UInt64}
end
@inline Base.pointer(rng::Xoshift) = rng.ptr

struct XoshiftState{P,W} <: AbstractState{P,W}
    eins::VecUnroll{P,W,UInt64,Vec{W,UInt64}}
    zwei::VecUnroll{P,W,UInt64,Vec{W,UInt64}}
    drei::VecUnroll{P,W,UInt64,Vec{W,UInt64}}
    vier::VecUnroll{P,W,UInt64,Vec{W,UInt64}}
end

Xoshift(ptr, ::StaticInt{X}) where {X} = Xoshift{X}(ptr)
Xoshift(ptr) = Xoshift(ptr, pick_vector_width(UInt64) * StaticInt(XREGISTERS))
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
        vstoreu!(ptr, e, 8i); vstoreu!(ptr, z, 8*(i + P)); vstoreu!(ptr, d, 8*(i + 2P)); vstoreu!(ptr, v, 8*(i + 3P));
        e, z, d, v = jump(e, z, d, v)
    end
    vstoreu!(ptr, e); vstoreu!(ptr, z, 8P); vstoreu!(ptr, d, 8*(2P)); vstoreu!(ptr, v, 8*(3P));
    vstoreu!(Base.unsafe_convert(Ptr{UInt32}, ptr), 0x00000000, 8*(4P));
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
    nstreams = XREGISTERS * Base.Threads.nthreads() * pick_vector_width(UInt64)
    initXoshift!(GLOBAL_vRNGs[], nstreams, e, z, d, v)
end


# @inline function getstate(rng::Xoshift{P}, ::Val{N}, ::Val{W}) where {P,N,W}
#     ptr = pointer(rng)
#     XoshiftState(
#         ntuple(n -> vloada(Vec{W,UInt64}, ptr, W64*(n - 1)), Val{N}()),
#         ntuple(n -> vloada(Vec{W,UInt64}, ptr, W64*(n - 1) + W64*P), Val{N}()),
#         ntuple(n -> vloada(Vec{W,UInt64}, ptr, W64*(n - 1) + 2W64*P), Val{N}()),
#         ntuple(n -> vloada(Vec{W,UInt64}, ptr, W64*(n - 1) + 3W64*P), Val{N}())
#     )
# end
@inline function getrandu64counter(rng::Xoshift{P}) where {P}
    vloadu(Base.unsafe_convert(Ptr{UInt8}, pointer(rng)), 4simd_integer_register_size()*P)
end
@inline function getrand64counter(rng::Xoshift{P}) where {P}
    vloadu(Base.unsafe_convert(Ptr{UInt8}, pointer(rng)), 4simd_integer_register_size()*P + 2)
end
@inline function getrandn64counter(rng::Xoshift{P}) where {P}
    vloadu(Base.unsafe_convert(Ptr{UInt8}, pointer(rng)), 4simd_integer_register_size()*P + 3)
end
@inline function setrandu64counter!(rng::Xoshift{P}, v::UInt8) where {P}
    vstoreu!(Base.unsafe_convert(Ptr{UInt8}, pointer(rng)), v, 4simd_integer_register_size()*P)
end
@inline function setrand64counter!(rng::Xoshift{P}, v::UInt8) where {P}
    vstoreu!(Base.unsafe_convert(Ptr{UInt8}, pointer(rng)), v, 4simd_integer_register_size()*P + 2)
end
@inline function setrandn64counter!(rng::Xoshift{P}, v::UInt8) where {P}
    vstoreu!(Base.unsafe_convert(Ptr{UInt8}, pointer(rng)), v, 4simd_integer_register_size()*P + 3)
end

@inline function getstate(rng::Xoshift{P}, ::Val{1}, ::StaticInt{W}) where {P,W}
    ptr = pointer(rng)
    XoshiftState(
        VecUnroll((vloada(ptr, MM{W,8}(StaticInt{0}())),)),
        VecUnroll((vloada(ptr, MM{W,8}(simd_integer_register_size()*P)),)),
        VecUnroll((vloada(ptr, MM{W,8}(2simd_integer_register_size()*P)),)),
        VecUnroll((vloada(ptr, MM{W,8}(3simd_integer_register_size()*P)),))
    )
end
@inline function getstate(rng::Xoshift{P}, ::Val{2}, ::StaticInt{W}) where {P,W}
    ptr = pointer(rng)
    XoshiftState(
        VecUnroll((vloada(ptr, MM{W,8}(StaticInt{0}()  )), vloada(ptr, MM{W,8}(simd_integer_register_size()         )))),
        VecUnroll((vloada(ptr, MM{W,8}( P*simd_integer_register_size())), vloada(ptr, MM{W,8}(simd_integer_register_size()*(1 +  P))))),
        VecUnroll((vloada(ptr, MM{W,8}(2P*simd_integer_register_size())), vloada(ptr, MM{W,8}(simd_integer_register_size()*(1 + 2P))))),
        VecUnroll((vloada(ptr, MM{W,8}(3P*simd_integer_register_size())), vloada(ptr, MM{W,8}(simd_integer_register_size()*(1 + 3P)))))
    )
end
@inline function getstate(rng::Xoshift{P}, ::Val{4}, ::StaticInt{W}) where {P,W}
    ptr = pointer(rng)
    RS = simd_integer_register_size()
    XoshiftState(
        VecUnroll((vloada(ptr, MM{W,8}(StaticInt{0}())), vloada(ptr, MM{W,8}(RS)         ), vloada(ptr, MM{W,8}(RS* 2)      ), vloada(ptr, MM{W,8}(RS* 3      )))),
        VecUnroll((vloada(ptr, MM{W,8}( P*RS)),          vloada(ptr, MM{W,8}(RS*(1 +  P))), vloada(ptr, MM{W,8}(RS*(2 +  P))), vloada(ptr, MM{W,8}(RS*(3 +  P))))),
        VecUnroll((vloada(ptr, MM{W,8}(2P*RS)),          vloada(ptr, MM{W,8}(RS*(1 + 2P))), vloada(ptr, MM{W,8}(RS*(2 + 2P))), vloada(ptr, MM{W,8}(RS*(3 + 2P))))),
        VecUnroll((vloada(ptr, MM{W,8}(3P*RS)),          vloada(ptr, MM{W,8}(RS*(1 + 3P))), vloada(ptr, MM{W,8}(RS*(2 + 3P))), vloada(ptr, MM{W,8}(RS*(3 + 3P)))))
    )
end
@inline function storestate!(rng::Xoshift{P}, s::XoshiftState{N,W}) where {P,N,W}
    ptr = pointer(rng)
    @unpack eins, zwei, drei, vier = s
    @inbounds for n ∈ 0:N
        vstorea!(rng, data(eins)[n], simd_integer_register_size()*n)
    end
    @inbounds for n ∈ 0:N
        vstorea!(rng, data(zwei)[n], simd_integer_register_size()*(n +  P))
    end
    @inbounds for n ∈ 0:N
        vstorea!(rng, data(drei)[n], simd_integer_register_size()*(n + 2P))
    end
    @inbounds for n ∈ 0:N
        vstorea!(rng, data(vier)[n], simd_integer_register_size()*(n + 3P))
    end
end
@inline function storestate!(rng::Xoshift{P}, s::XoshiftState{0,W}) where {P,W}
    ptr = pointer(rng)
    @unpack eins, zwei, drei, vier = s;
    _eins = data(eins); _zwei = data(zwei); _drei = data(drei); _vier = data(vier);
    @inbounds begin
        vstorea!(ptr, _eins[1],       )
        vstorea!(ptr, _zwei[1],  P*simd_integer_register_size())
        vstorea!(ptr, _drei[1], 2P*simd_integer_register_size())
        vstorea!(ptr, _vier[1], 3P*simd_integer_register_size())
    end
end
@inline function storestate!(rng::Xoshift{P}, s::XoshiftState{1,W}) where {P,W}
    ptr = pointer(rng)
    @unpack eins, zwei, drei, vier = s;
    _eins = data(eins); _zwei = data(zwei); _drei = data(drei); _vier = data(vier);
    @inbounds begin
        vstorea!(ptr, _eins[1],                            )
        vstorea!(ptr, _eins[2],   simd_integer_register_size()          )
        vstorea!(ptr, _zwei[1],   simd_integer_register_size()*       P )
        vstorea!(ptr, _zwei[2],   simd_integer_register_size()*(1 +   P))
        vstorea!(ptr, _drei[1],   simd_integer_register_size()*      2P )
        vstorea!(ptr, _drei[2],   simd_integer_register_size()*(1 +  2P))
        vstorea!(ptr, _vier[1],   simd_integer_register_size()*      3P )
        vstorea!(ptr, _vier[2],   simd_integer_register_size()*(1 +  3P))
    end
end
@inline function storestate!(rng::Xoshift{P}, s::XoshiftState{3,W}) where {P,W}
    ptr = pointer(rng)
    @unpack eins, zwei, drei, vier = s;
    _eins = data(eins); _zwei = data(zwei); _drei = data(drei); _vier = data(vier);
    @inbounds begin
        vstorea!(ptr, _eins[1],      )
        vstorea!(ptr, _eins[2],   simd_integer_register_size())
        vstorea!(ptr, _eins[3],   simd_integer_register_size()*2)
        vstorea!(ptr, _eins[4],   simd_integer_register_size()*3)
        vstorea!(ptr, _zwei[1],   simd_integer_register_size()*       P)
        vstorea!(ptr, _zwei[2],   simd_integer_register_size()*(1 +   P))
        vstorea!(ptr, _zwei[3],   simd_integer_register_size()*(2 +   P))
        vstorea!(ptr, _zwei[4],   simd_integer_register_size()*(3 +   P))
        vstorea!(ptr, _drei[1],   simd_integer_register_size()*      2P)
        vstorea!(ptr, _drei[2],   simd_integer_register_size()*(1 +  2P))
        vstorea!(ptr, _drei[3],   simd_integer_register_size()*(2 +  2P))
        vstorea!(ptr, _drei[4],   simd_integer_register_size()*(3 +  2P))
        vstorea!(ptr, _vier[1],   simd_integer_register_size()*      3P)
        vstorea!(ptr, _vier[2],   simd_integer_register_size()*(1 +  3P))
        vstorea!(ptr, _vier[3],   simd_integer_register_size()*(2 +  3P))
        vstorea!(ptr, _vier[4],   simd_integer_register_size()*(3 +  3P))
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

@generated function _unpack(s::XoshiftState{P}, ::Val{U}) where {P,U}
    if U ≤ P
        quote
            $(Expr(:meta,:inline))
            @unpack eins, zwei, drei, vier = s
            _eins = data(eins); _zwei = data(zwei); _drei = data(drei); _vier = data(vier);
            (
                VecUnroll( Base.Cartesian.@ntuple $U u -> _eins[u] ),
                VecUnroll( Base.Cartesian.@ntuple $U u -> _zwei[u] ),
                VecUnroll( Base.Cartesian.@ntuple $U u -> _drei[u] ),
                VecUnroll( Base.Cartesian.@ntuple $U u -> _vier[u] )
            )
        end
    elseif U == P+1
        quote
            $(Expr(:meta,:inline))
            s.eins, s.zwei, s.drei, s.vier
        end
    else
        throw("$U > $(P+1).")
    end
end

@inline XoshiftState(eins::VecUnroll{N}, zwei::VecUnroll{N}, drei::VecUnroll{N}, vier::VecUnroll{N}, s::XoshiftState{N}) where {N} = XoshiftState( eins, zwei, drei, vier )
@generated function XoshiftState(eins::VecUnroll{N}, zwei::VecUnroll{N}, drei::VecUnroll{N}, vier::VecUnroll{N}, s::XoshiftState{P}) where {N,P}
    @assert P > N
    q = quote
        $(Expr(:meta,:inline))
        e = data(eins); z = data(zwei); d = data(drei); v = data(vier)
        _e = data(s.eins); _z = data(s.zwei); _d = data(s.drei); _v = data(s.vier)
    end
    _eins = Expr(:tuple)
    _zwei = Expr(:tuple)
    _drei = Expr(:tuple)
    _vier = Expr(:tuple)
    for n ∈ 1:N+1
        push!(_eins.args, Expr(:ref, :e, n))
        push!(_zwei.args, Expr(:ref, :z, n))
        push!(_drei.args, Expr(:ref, :d, n))
        push!(_vier.args, Expr(:ref, :v, n))
    end
    for n ∈ N+2:P+1
        push!(_eins.args, Expr(:ref, :_e, n))
        push!(_zwei.args, Expr(:ref, :_z, n))
        push!(_drei.args, Expr(:ref, :_d, n))
        push!(_vier.args, Expr(:ref, :_v, n))
    end
    push!(q.args, :(XoshiftState( VecUnroll($_eins), VecUnroll($_zwei), VecUnroll($_drei), VecUnroll($_vier) )))
    q
end

@inline function nextstate(s::XoshiftState{P}, ::Val{U}) where {P,U}
    eins, zwei, drei, vier = _unpack(s, Val{U}())
    out = eins + vier
    out = rotate_right(out, 0x0000000000000017)
    eins, zwei, drei, vier = nextstate(eins, zwei, drei, vier)
    # out += eins
    XoshiftState( eins, zwei, drei, vier, s ), out
end

function randbuffer64(r::Xoshift{P}) where {P}
    ptr = pointer(r)
    Buffer256(Base.unsafe_convert(Ptr{Float64}, ptr + P * 5simd_integer_register_size()))
end
function randnbuffer64(r::Xoshift{P}) where {P}
    ptr = pointer(r)
    Buffer256(Base.unsafe_convert(Ptr{Float64}, ptr + P * 5simd_integer_register_size() + 2048))
end
# function randbuffer32(r::Xoshift{P}) where {P}
#     ptr = pointer(r)
#     Buffer256(Base.unsafe_convert(Ptr{Float64}, ptr + P * 5simd_integer_register_size() + 4096))
# end
function randubuffer64(r::Xoshift{P}) where {P}
    ptr = pointer(r)
    Buffer256(Base.unsafe_convert(Ptr{UInt64}, ptr + P * 5simd_integer_register_size() + 4096))
end
# function randnbuffer32(r::Xoshift{P}) where {P}
#     ptr = pointer(r)
#     Buffer256(Base.unsafe_convert(Ptr{Float64}, ptr + P * 5simd_integer_register_size() + 5120))
# end

