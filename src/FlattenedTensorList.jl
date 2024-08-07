
module FlattenedTensorListModule

struct FTLPositionInfo{IXT,N}
    offset::IXT
    len::IXT
    shape::NTuple{N,IXT}
    strides::NTuple{N,IXT}
end
# const FTLPositionInfo = Tuple{IXT,IXT,NTuple{N,IXT},NTuple{N,IXT}} where {IXT,N}


# The FlattenedTensorList stores a flattened representation of a list of list of tensors.
# The first axis is the batch axis. The second axis is the feature axis.
# Different features can have diferent shapes, but along the batch axes, the shapes are the same.
# The flattened representation also stores the batch axis last, so it shouldn't have problems with cache.
struct FlattenedTensorList{T,N,IXT,AT<:AbstractVector{T},APosInfo<:AbstractVector{FTLPositionInfo{IXT,N}}}
    B::IXT # number of samples
    L::IXT # total length in Ts of each sample
    flattened::AT # A B*L array of Ts
    positions::APosInfo
    # for each feature:
    #   the index into the flattened array at which it starts at
    #   the length of the feature
    #   N sizes representing the size of the tensor
    #   N strides
end

# This might be faster than using views, which somehow cause copying
struct TensorIndex{IXT,N,T,AT<:AbstractVector{T}}
    ar::AT
    offset::IXT
    shape::NTuple{N,IXT}
    strides::NTuple{N,IXT}
end

@inline function continuous_dims(tix::TensorIndex{IXT,N}) where {IXT,N}
    acum = 1
    for nx in 1:N
        if acum != tix.strides[nx]
            return nx-1
        end
        acum *= tix.shape[nx]
    end
    return N
end

@inline function continuous_dims(tix::TensorIndex{IXT,N}, tixs::Vararg{TensorIndex{IXT,N},K}) where {IXT,N,K}
    acum = 1
    for nx in 1:N
        if acum != tix.strides[nx]
            return nx-1
        end
        for kx in 1:K
            if tixs[kx].strides[nx] != tix.strides[nx] || tixs[kx].shape[nx] != tix.shape[nx]
                return nx-1
            end
        end
        acum *= tix.shape[nx]
    end
    return N
end

# @inline function merge_continuous_dims(tix::TensorIndex{IXT,N}, ::Val{cd}) where {IXT,N,cd}
#     # ::TensorIndex{IXT,N-cd}
#     # if it's zero it's the original array
#     # if it's one it's still the original array?
#     # if it's 2, we merge the first two
#     return TensorIndex{IXT,N-cd}(
#         tix.ar, tix.offset, 
#         (prod(ntuple(i -> tix.shape[i], Val(cd+1))), ntuple(i -> tix.shape[i+cd], Val(N-cd-1))...),
#         tuple(i -> tix.shape[i+cd], Val(N-cd))
#     )
# end

@inline Base.axes(ti::TensorIndex, dim::Integer) = 1:(ti.shape[dim])

@inline function ti_from_array(x::AT) where {T,N,AT<:AbstractArray{T,N}}
    return TensorIndex{Int32,N,T,typeof(x[:])}(copy(@view(x[:])), 0, size(x), begin
        acum = 1
        ntuple(i -> i == 1 ? 1 : (acum *= size(x, i-1)), Val(N))
    end)
end

@inline function feature(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer) where {T,N,IXT,AT}
    return TensorIndex{IXT,N+1,T,AT}(ftl.flattened, ftl.positions[fi].offset, (ftl.positions[fi].shape..., ftl.B), (ftl.positions[fi].strides..., ftl.L))
end

@inline function Base.length(tix::TensorIndex{IXT,N}) where {IXT,N}
    return prod(tix.shape)
end

@inline function feature_flat(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer) where {T,N,IXT,AT}
    return TensorIndex{IXT,2,T,AT}(ftl.flattened, ftl.positions[fi].offset, (ftl.positions[fi].len, ftl.B), (1, ftl.L))
end

@inline function feature_flat(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer, bi::Integer) where {T,N,IXT,AT}
    return TensorIndex{IXT,1,T,AT}(ftl.flattened, ftl.L*(bi-1) + ftl.positions[fi].offset, (ftl.positions[fi].len,), (1,))
end

function feature(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer, bi::Integer) where {T,N,IXT,AT}
    return TensorIndex{IXT,N,T,AT}(ftl.flattened, ftl.positions[fi].offset + ftl.L * convert(IXT, bi-1), ftl.positions[fi].shape, ftl.positions[fi].strides)
end

@inline function feature(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer, bis::UnitRange) where {T,N,IXT,AT}
    return TensorIndex{IXT,N+1,T,AT}(ftl.flattened, convert(IXT, ftl.positions[fi].offset + ftl.L*(bis.start-1)), (ftl.positions[fi].shape..., convert(IXT, bis.stop-bis.start+1)), (ftl.positions[fi].strides..., ftl.L))
end

@inline function sample_flat(ftl::FlattenedTensorList{T,N,IXT,AT}, bi::Integer) where {T,N,IXT,AT}
    return TensorIndex{IXT,1,T,AT}(ftl.flattened, (bi-1)*ftl.L, (ftl.L,), (1,))
end

@inline function sample_flat(ftl::FlattenedTensorList{T,N,IXT,AT}, bis::UnitRange) where {T,N,IXT,AT}
    return TensorIndex{IXT,2,T,AT}(ftl.flattened, (bis.start-1)*ftl.L, (ftl.L,bis.stop-bis.start+1), (1,ftl.L))
end

@inline function value(tix::TensorIndex{IXT,N,T}, ixs::Vararg{Integer,N}) where {T,IXT,N}
    return  tix.ar[sum(ntuple(nx -> (ixs[nx]-1) * tix.strides[nx], Val(N))) + tix.offset + 1]
end

@inline function indices(tix::TensorIndex{IXT,N,T}, i::Integer) where {T,IXT,N}
    acum = i-1
    return ntuple(si -> begin
        v = acum % tix.shape[si]
        acum = div(acum, tix.shape[si])
        return v+1
    end, Val(N))
end

@inline function index(tix::TensorIndex{IXT,N,T}, ixs::Vararg{IXT, N}) where {T,IXT,N}
    return sum(ntuple(nx -> (ixs[nx]-1) * tix.strides[nx], Val(N))) + tix.offset + 1
end

@inline function index_clamp(tix::TensorIndex{IXT,N,T}, ixs::Vararg{IXT, N}) where {T,IXT,N}
    return sum(ntuple(nx -> ixs[nx] > tix.shape[nx] ? 0 : (ixs[nx]-1) * tix.strides[nx], Val(N))) + tix.offset + 1
end

@inline function selectdim_ti(tix::TensorIndex{IXT,N,T}, dim::Integer, i::Integer) where {T,IXT,N}
    return TensorIndex(tix.ar, tix.offset + convert(IXT, i-1)*tix.strides[dim], ntuple(nx -> nx == dim ? one(IXT) : tix.shape[nx], Val(N)), tix.strides)
end

@inline function selectdim_ti(tix::TensorIndex{IXT,N,T}, dim::Integer, ir::UnitRange{<:Integer}) where {T,IXT,N}
    return TensorIndex(tix.ar, 
    tix.offset + convert(IXT, ir.start-1)*tix.strides[dim], 
    ntuple(nx -> nx == dim ? convert(IXT, ir.stop - ir.start + 1) : tix.shape[nx], Val(N)), 
    tix.strides)
end

using StaticArrays


function _map1_ti!(op::F, dest::TensorIndex{IXT,N}) where{F,IXT,N}
    ii = MVector{N,IXT}(ntuple(Returns(1), Val(N)))
    iterlen = length(dest)
    dinar = dest.offset+1
    djumps = MVector{N,IXT}(undef)
    djumps[1] = dest.strides[1]
    for nx in 2:N
        djumps[nx] = dest.strides[nx] - dest.shape[nx-1]*dest.strides[nx-1]
    end
    i = 0
    while true
        i += 1

        dest.ar[dinar] = op((@inbounds dest.ar[dinar]))

        if i >= iterlen
            break
        end

        ii[1] += 1
        dinar += djumps[1]

        nx = 1
        while (@inbounds ii[nx]) > (@inbounds dest.shape[nx])
            @inbounds ii[nx+1] += 1
            @inbounds ii[nx] = 1
            dinar += @inbounds djumps[nx+1]
            nx+=1
        end
        
    end
    return nothing
end

function _map1v_ti!(op::F, dest::TensorIndex{IXT,N}, ::Val{V}) where{F,IXT,N,V}
    ii = MVector{N-V,IXT}(ntuple(Returns(1), Val(N-V)))
    iterlen = 1
    viewlen = 1
    for i in 1:V
        viewlen *= dest.shape[i]
    end
    viewlen-=1
    for i in (V+1):N
        iterlen *= dest.shape[i]
    end
    #@show dest.shape, dest.strides, V, iterlen, viewlen
    dinar = dest.offset+1
    djumps = MVector{N-V,IXT}(undef)
    djumps[1] = dest.strides[V+1]
    for nx in 2:(N-V)
        djumps[nx] = dest.strides[V+nx] - dest.shape[V+nx-1]*dest.strides[V+nx-1]
    end
    i = 0

    while true
        i += 1

        for j in 0:viewlen
            @inbounds dest.ar[dinar+j] = op((@inbounds dest.ar[dinar+j]))
        end

        if i >= iterlen
            break
        end

        ii[1] += 1
        dinar += djumps[1]

        nx = 1
        while (@inbounds ii[nx]) > (@inbounds dest.shape[V+nx])
            @inbounds ii[nx+1] += 1
            @inbounds ii[nx] = 1
            dinar += @inbounds djumps[nx+1]
            nx+=1
        end
        
    end
    return nothing
end

function _mapkv_ti!(op::F, dest::TensorIndex{IXT,N}, ::Val{V}, sources::Vararg{TensorIndex{IXT,N}, K}) where {V,F,IXT,N,K}
    
    itershape = ntuple(nx -> max(dest.shape[V+nx], maximum(source -> source.shape[V+nx], sources; init=0)), Val(N-V))
    # asserts
    # remove once the function is verified to work
    # for nx in 1:N
    #     @assert itershape[nx] == dest.shape[nx] || dest.shape[nx] == 1
    #     for kx in 1:K
    #         @assert itershape[nx] == sources[kx].shape[nx] || sources[kx].shape[nx] == 1
    #     end 
    # end
    # println("mapk_ti!")
    viewlen = 1
    for i in 1:V
        viewlen *= dest.shape[i]
    end
    viewlen -= 1

    iterlen = prod(itershape)
    ii = MVector{N-V,IXT}(ntuple(Returns(1), Val(N-V)))
    # why can't julia have static mutable arrays on the stack??
    # we will have to do this janky thing
    
    sinar = MVector{K,IXT}(ntuple(kx -> sources[kx].offset+1, Val(K)))
    dinar = dest.offset+1

    djumps = MVector{N-V,IXT}(undef)
    sjumps = MMatrix{K,N-V,Int32}(undef)
    # xinar = xi[1] * strides[1] + xi[2] * strides[2] + xi[3] * strides[3] + ....
    # xi[1] = if shape[1] == itershape[1] ? 1 : ii[1]
    # djumps[nx] - what change does occur to dinar when we increment di[nx]
    # we are in the equal case, we increment xi, then it increases by strides[nx]
    djumps[1] = dest.shape[V+1] == itershape[1] ? dest.strides[V+1] : 0
    for kx in 1:K
        sjumps[kx,1] = sources[kx].shape[V+1] == itershape[1] ? sources[kx].strides[V+1] : 0
    end
    for nx in 2:(N-V)
        djumps[nx] = (dest.shape[V+nx] == itershape[nx] ? dest.strides[V+nx] : 0) - (dest.shape[V+nx-1] == itershape[nx-1] ? dest.shape[V+nx-1]*dest.strides[V+nx-1] : 0)
        for kx in 1:K
            sjumps[kx,nx] = (sources[kx].shape[V+nx] == itershape[nx] ? sources[kx].strides[V+nx] : 0) - (sources[kx].shape[V+nx-1] == itershape[nx-1] ? sources[kx].shape[V+nx-1]*sources[kx].strides[V+nx-1] : 0)
        end
    end
    #display(dest)
    #for kx in 1:K display(sources[kx]) end
    #@show itershape, djumps, sjumps

    i = 0
    while true
        i += 1
        #@show dinar, sinar, ii
        for j in 0:viewlen
            @inbounds dest.ar[dinar+j] = op(
                (@inbounds dest.ar[dinar+j]),
                ntuple(kx -> (@inbounds sources[kx].ar[sinar[kx]+j]), Val(K))...
            )
        end

        if i >= iterlen
            break
        end

        ii[1] += 1
        dinar += djumps[1]
        for kx in 1:K
            @inbounds sinar[kx] += @inbounds sjumps[kx,1]
        end

        nx = 1
        while (@inbounds ii[nx]) > (@inbounds itershape[nx])
            @inbounds ii[nx+1] += 1
            @inbounds ii[nx] = 1
            dinar += @inbounds djumps[nx+1]
            for kx in 1:K
                sinar[kx] += @inbounds sjumps[kx,nx+1]
            end
            nx+=1
        end
        
        # TODO: remove asserts later after veryfing that this function always works
        # @show dinar, sinar, ii
        # @assert dinar == index_clamp(dest, ii...)
        # for kx in 1:K
        #     @assert sinar[kx] == index_clamp(sources[kx], ii...)
        # end
        
    end
    return nothing

end

function _mapk_ti!(op::F, dest::TensorIndex{IXT,N}, sources::Vararg{TensorIndex{IXT,N}, K}) where {F,IXT,N,K}
    
    itershape = ntuple(nx -> max(dest.shape[nx], maximum(source -> source.shape[nx], sources; init=0)), Val(N))

    iterlen = prod(itershape)
    ii = MVector{N,IXT}(ntuple(Returns(1), Val(N)))
    # why can't julia have static mutable arrays on the stack??
    # we will have to do this janky thing
    
    sinar = MVector{K,IXT}(ntuple(kx -> sources[kx].offset+1, Val(K)))
    dinar = dest.offset+1

    djumps = MVector{N,IXT}(undef)
    sjumps = MMatrix{K,N,Int32}(undef)
    # xinar = xi[1] * strides[1] + xi[2] * strides[2] + xi[3] * strides[3] + ....
    # xi[1] = if shape[1] == itershape[1] ? 1 : ii[1]
    # djumps[nx] - what change does occur to dinar when we increment di[nx]
    # we are in the equal case, we increment xi, then it increases by strides[nx]
    djumps[1] = dest.shape[1] == itershape[1] ? dest.strides[1] : 0
    for kx in 1:K
        sjumps[kx,1] = sources[kx].shape[1] == itershape[1] ? sources[kx].strides[1] : 0
    end
    for nx in 2:N
        djumps[nx] = (dest.shape[nx] == itershape[nx] ? dest.strides[nx] : 0) - (dest.shape[nx-1] == itershape[nx-1] ? dest.shape[nx-1]*dest.strides[nx-1] : 0)
        for kx in 1:K
            @inbounds sjumps[kx,nx] = (sources[kx].shape[nx] == itershape[nx] ? sources[kx].strides[nx] : 0) - (sources[kx].shape[nx-1] == itershape[nx-1] ? sources[kx].shape[nx-1]*sources[kx].strides[nx-1] : 0)
        end
    end
    #display(dest)
    #for kx in 1:K display(sources[kx]) end
    #@show itershape, djumps, sjumps

    i = 0
    while true
        i += 1
        #@show dinar, sinar, ii

        @inbounds dest.ar[dinar] = op(
            @inbounds(dest.ar[dinar]),
            ntuple(kx -> @inbounds(sources[kx].ar[sinar[kx]]), Val(K))...
        )

        if i >= iterlen
            break
        end

        ii[1] += 1
        dinar += djumps[1]
        for kx in 1:K
            sinar[kx] += @inbounds(sjumps[kx,1])
        end

        nx = 1
        while @inbounds(ii[nx]) > @inbounds(itershape[nx])
            @inbounds ii[nx+1] += 1
            @inbounds ii[nx] = 1
            dinar += @inbounds(djumps[nx+1])
            for kx in 1:K
                @inbounds sinar[kx] += @inbounds(sjumps[kx,nx+1])
            end
            nx+=1
        end
        
        # TODO: remove asserts later after veryfing that this function always works
        # @show dinar, sinar, ii
        # @assert dinar == index_clamp(dest, ii...)
        # for kx in 1:K
        #     @assert sinar[kx] == index_clamp(sources[kx], ii...)
        # end
        
    end
    return nothing

end

@inline function _mapk_full_ti!(op::F, dest::TensorIndex{IXT,N}, sources::Vararg{TensorIndex{IXT,N},K}) where {F,IXT,N,K}
    for j in 1:length(dest)
        @inbounds dest.ar[dest.offset+j] = op(@inbounds(dest.ar[dest.offset+j]), ntuple(kx -> (@inbounds sources[kx].ar[sources[kx].offset+j]), Val(K))...)
    end
    return nothing
end

@generated function mapk_ti!(op::F, dest::TensorIndex{IXT,N}, sources::Vararg{TensorIndex{IXT,N},K}) where {F,IXT,N,K}
    #_mapk_ti!(op, dest, sources...)
    return if K == 0 quote 
        cd = continuous_dims(dest)
        if cd == 0
            # println("_map1_ti!")
            _map1_ti!(op, dest)
        elseif cd == N
            # println("_map1_full_ti!")
            for k in 1:length(dest)
                @inbounds dest.ar[dest.offset+k] = op(@inbounds dest.ar[dest.offset+k])
            end
        else
            # println("_map1v_ti!")
            Base.@nif($N, i -> cd == i, i -> _map1v_ti!(op, dest, Val(i)))
        end
        return nothing
    end else quote
        cd = continuous_dims(dest, sources...)
        if cd == 0
            # println("_mapk_ti!")
            _mapk_ti!(op, dest, sources...)
        elseif cd == N
            # println("_mapk_full_ti!")
            _mapk_full_ti!(op, dest, sources...)
        else
            # println("_mapkv_ti!")
            Base.@nif($N, i -> cd == i, i -> _mapkv_ti!(op, dest, Val(i), sources...))
        end
        return nothing
    end end
end

@inline function zero_ti!(dest::TensorIndex{IXT,N}) where{IXT,N}
    mapk_ti!(Returns(zero(IXT)), dest)
end

@inline function copyto_ti!(dest::TensorIndex{IXT,N}, src::TensorIndex{IXT,N}) where{IXT,N}
    mapk_ti!((_,s) -> s, dest, src)
end

@inline function addto_ti!(op::F, dest::TensorIndex{IXT,N}, sources::Vararg{TensorIndex{IXT,N},K}) where{F,IXT,N,K}
    mapk_ti!((dt,sts...) -> dt+op(sts...), dest, sources...)
end

@inline function addto_ti!(dest::TensorIndex{IXT,N}, source::TensorIndex{IXT,N}) where{IXT,N}
    mapk_ti!((dt,st) -> dt+st, dest, source)
end


# using a 100_000x5x5x5 array: 
#                without op  with op 
# new copy        ~14.5 ms    ~16.0ms    copy_ti!(feature(ftl, 1), feature(ftl, 2))
# new copy flat   ~14.4 ms    ~14.6ms    copy_ti!(feature_flat(ftl, 1), feature_flat(ftl, 2))
# new copy simple ~14ms       ~14ms      copy_ti_simple!(op, feature_flat(ftl, 1), feature_flat(ftl, 2))
# raw vectors     ~5.8 ms     ~9.6ms     x1 .= x2
# 1-dim views     ~9.4 ms     ~9.4ms     @view(x1[:]) .= @view(x1[:]
# N-dim views     ~14.5 ms    ~7.8ms     @view(x1[1:B,:,:,:]) .= @view(x1[1:B,:,:,:]
# before          ~1.7s       ~660ms     ftl[1] .= ftl[2] (due to copying somewhere)
# mapk_ti!        ~23.2ms or ~15ms (they sometimes give diffrent reuslts even using @benchmark)
# mapk_ti! flat   ~15.3ms or ~22ms (they sometimes give diffrent reuslts even using @benchmark)
#function copy_ti!(op::F, dest::TensorIndex{IXT,N}, source::TensorIndex{IXT,N}) where {IXT, N, F}

# Benchmarks for 100_000:
# first version: 15.8ms
# second version: 15.0mms
# raw arrays: 2.5ms
# views:      2.5ms
# before:     220ms
# still a 15x improvement, but not compared to the 80x improvement for raw arrays
#function _map2b_ti!(op::F, dest::TensorIndex{IXT,N}, lterm::TensorIndex{IXT,N}, rterm::TensorIndex{IXT,N}) where {IXT,N,F}

# for 100_000x10x10:
# _map2nbv_ti! 1ms
# raw arrays   9ms
# views        10ms
# function _map2nbv_ti!(op::F, dest::TensorIndex{IXT,N}, lterm::TensorIndex{IXT,N}, rterm::TensorIndex{IXT,N}, viewdims::Integer) where {IXT,N,F}

# zeroing out a 3000 vector
# x .= 0   -> 80ns
# x .= Returns(0).(x) -> 110 ns
# mapself_ti!(Returns(0), f) -> 33ns (how is this faster?)
#function mapself_ti!(op::F, dest::TensorIndex{IXT,N}) where {F,IXT,N}

function materialize_ti(source::TensorIndex{IXT,N,T,AT}) where {IXT,N,T,AT}
    v = Array{T,N}(undef, source.shape...)
    l = length(source)
    si = MVector{N,IXT}(ntuple(Returns(1), Val(N))...)
    si[1] = 0
    sinar = source.offset
    for _ in 1:l
        sinar += source.strides[1]
        si[1] += 1
        for nx in 1:N
            if si[nx] > source.shape[nx]
                si[nx+1] += 1
                si[nx] = 1
                sinar +=  source.strides[nx+1] - source.shape[nx]*source.strides[nx]
            end
        end
         v[si...] =  source.ar[sinar]
    end
    return v
end

function treat_as_flattened(buff::AT, sizes::AbstractVector{NTuple{N, IXT}}, max_B::Integer) where {T,IXT,N,AT<:AbstractVector{T}}
    positions = Vector{FTLPositionInfo{IXT,N}}(undef, length(sizes))

    acum = 0
    for fi in eachindex(sizes)
        positions[fi] = FTLPositionInfo{IXT,N}(acum, prod(sizes[fi]), sizes[fi], (1, cumprod(Base.front(sizes[fi]))...))
        acum += positions[fi].len
    end
    l = positions[end].offset + positions[end].len
    b = div(length(buff), l)
    b = min(b, max_B)
    return FlattenedTensorList{T,N,IXT,AT,Vector{FTLPositionInfo{IXT,N}}}(b, l, buff, positions)
end

function flatten(::Type{AT}, X::AbstractVector{<:AbstractArray{T,NP1}}; index_type::Val{IXT}=Val(Int32)) where {IXT,T,NP1,AT<:AbstractArray}
    N = NP1-1
    B = size(X[1], 1)
    l = sum(Xi -> div(length(Xi), B), X)
    f = length(X)
    flattened = Vector{T}(undef, B*l)
    positions = Vector{FTLPositionInfo{IXT,N}}(undef, f)
    acum = 0
    for fi in 1:f
        positions[fi] = FTLPositionInfo{IXT,N}(acum, div(length(X[fi]), B), Base.tail(size(X[fi])), div.(Base.tail(strides(X[fi])),B))
        acum += positions[fi].len
    end
    for bi in 1:B, fi in 1:f
        acum = positions[fi].offset
        fl = positions[fi].len
        @view(flattened[(l*(bi-1)+acum+1):(l*(bi-1)+acum+fl)]) .= reshape(selectdim(X[fi], 1, bi), (fl,))
    end
    return FlattenedTensorList{T, N, IXT, AT, Vector{FTLPositionInfo{IXT, N}}}(B, l, if AT == Vector{T} flattened else AT(flattened) end, positions)
end

function permute_features(ftl::FlattenedTensorList{T,N,IXT,AT,APT}, features::AbstractVector{IXT}) where {T,N,IXT,AT,APT}
    B = ftl.B
    l = sum(idx -> ftl.positions[idx].len, features)
    reordered = AT(undef, B*l)
    positions = APT(undef, length(features))
    position = zero(IXT)
    for i in eachindex(features)
        positions[i] = FTLPositionInfo(position, ftl.positions[features[i]].len, ftl.positions[features[i]].shape, ftl.positions[features[i]].strides)
        position += positions[i].len
    end
    newftl = FlattenedTensorList{T,N,IXT,AT,APT}(B, l, reordered, positions)
    for i in eachindex(features)
        copyto_ti!(feature_flat(newftl, i), feature_flat(ftl, features[i]))
        # @view(reshape(@view(reordered[:]), (l,B))[(position+1):(position+size), :]) .= 
        #     @view(reshape(@view(ftl.flattened[:]), (ftl.L, ftl.B))[(old_pos+1):(old_pos+size), :])
    end
    return newftl
end

# function permute_features!(ftl::FlattenedTensorList, features::AbstractVector)
#     ftl2 = permute_features(ftl, features)
#     ftl.flattened = ftl2.flattened
#     ftl.positions = ftl2.positions
#     ftl.L = ftl2.L
#     ftl.B = ftl2.B
#     return ftl
# end

# # gets an element
# @inline function Base.getindex(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer, bi::Integer, ixs::Vararg{Integer,N}) where {T,N,IXT,AT}
#     acum = ftl.positions[fi].offset
#     stride = ftl.positions[fi].strides
#     fix = sum(NTuple{N,IXT}(ixs) .* (stride .- one(IXT))) + one(IXT)
#     return ftl.flattened[ftl.L*(bi-1) + acum + fix]
# end

# # returns a reshape(view(AbstractArray)) representing a feature with all the samples
# @inline function Base.getindex(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer) where {T,N,IXT,AT}
#     acum = ftl.positions[fi].offset
#     fl = ftl.positions[fi].len
#     sizes = ftl.positions[fi].shape
#     ffv = @view(reshape(@view(ftl.flattened[1:(ftl.L*ftl.B)]), (ftl.L, ftl.B))[(acum + 1):(acum + fl), :])
#     return reshape(ffv, (sizes..., ftl.B))
# end

# # returns a reshape(view(AbstractArray)) representing a feature of the given sample
# @inline function Base.getindex(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer, bi::Integer) where {T,N,IXT,AT}
#     #acum, fl, sizes, _ = ftl.positions[fi]
#     acum, = ftl.positions[fi].offset
#     fl = ftl.positions[fi].len
#     sizes = ftl.positions[fi].shape
#     ffv = @view(reshape(@view(ftl.flattened[1:(ftl.L*ftl.B)]), (ftl.L, ftl.B))[(acum + 1):(acum + fl), bi])
#     return reshape(ffv, sizes)
# end

function Base.display(ftl::FlattenedTensorList{T,N,IXT,AT}) where {T,N,IXT,AT}
    print("$(length(ftl.positions))-feature $(ftl.B)-sample $(ftl.L)-samplelength FlattenedTensorList{$(T),$(N)} of sizes:\n")
    for i in eachindex(ftl.positions)
        print("   $(ftl.positions[i].shape) at pos $(ftl.positions[i].offset)\n")
    end
    # println("with values:")
    # for i in eachindex(ftl.positions)
    #     print("   ", ftl[i, 1], "\n")
    # end
end

function Base.display(ix::TensorIndex{IXT,N}) where {IXT,N}
    print("TensorIndex{$(IXT),$(N)} at offset $(ix.offset) with size $(ix.shape) and strides $(ix.strides)\n")
end

end