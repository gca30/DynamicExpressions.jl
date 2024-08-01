
module FlattenedTensorListModule

mutable struct FTLPositionInfo{IXT,N}
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
mutable struct FlattenedTensorList{T,N,IXT,AT<:AbstractVector{T},APosInfo<:AbstractVector{FTLPositionInfo{IXT,N}}}
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

mutable struct TensorIndex{IXT,N,T,AT<:AbstractVector{T}}
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

@inline function merge_continuous_dims(tix::TensorIndex{IXT,N}, ::Val{cd}) where {IXT,N,cd}
    # ::TensorIndex{IXT,N-cd}
    # if it's zero it's the original array
    # if it's one it's still the original array?
    # if it's 2, we merge the first two
    return TensorIndex{IXT,N-cd}(
        tix.ar, tix.offset, 
        (prod(ntuple(i -> tix.shape[i], Val(cd+1))), ntuple(i -> tix.shape[i+cd], Val(N-cd-1))...),
        tuple(i -> tix.shape[i+cd], Val(N-cd))
    )
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

@inline function feature_sample(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer, bi::Integer) where {T,N,IXT,AT}
    return TensorIndex{IXT,N,T,AT}(ftl.flattened, ftl.positions[fi].offset + ftl.L*(bi-1), ftl.positions[fi].shape, ftl.positions[fi].strides)
end

@inline function feature(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer, bis::UnitRange) where {T,N,IXT,AT}
    return TensorIndex{IXT,N,T,AT}(ftl.flattened, ftl.positions[fi].offset + ftl.L*(bis.start-1), (ftl.positions[fi].shape..., bis.stop-bis.start+1), (ftl.positions[fi].strides..., ftl.L))
end

@inline function value(tix::TensorIndex{IXT,N,T}, ixs::Vararg{Integer,N}) where {T,IXT,N}
    return @inbounds tix.ar[prod(ntuple(nx -> (ixs[nx]-1) * tix.strides[nx], Val(N))) + offset]
end

@inline function indices(tix::TensorIndex{IXT,N,T}, i::Integer) where {T,IXT,N}
    acum = i-1
    return ntuple(si -> begin
        v = acum % tix.shape[si]
        acum = div(acum, tix.shape[si])
        return v+1
    end, Val(N))
end

function copy_ti_simple!(dest::TensorIndex{IXT,N}, source::TensorIndex{IXT,N}) where {IXT,N}
    l = length(dest)
    di = MVector{N,IXT}(ntuple(Returns(1), Val(N))...)
    doffjump = ntuple(nx -> dest.strides[nx+1] - dest.shape[nx]*dest.strides[nx], Val(N-1))
    soffjump = ntuple(nx -> source.strides[nx+1] - source.shape[nx]*source.strides[nx], Val(N-1))
    di[1] = 0
    dinar = dest.offset
    sinar = source.offset
    i = 0
    while i < l
        i += 1
        dinar += dest.strides[1]
        sinar += source.strides[1]
        di[1] += 1
        nx = 1
        while di[nx] > dest.shape[nx]
            di[nx+1] += 1
            di[nx] = 1
            dinar += @inbounds doffjump[nx]
            sinar += @inbounds soffjump[nx]
            nx += 1
        end
        @inbounds dest.ar[dinar] = @inbounds source.ar[sinar]
    end
    return nothing
end

function copy_ti_views(dest::TensorIndex{IXT,N}, source::TensorIndex{IXT,N}, viewdims::Integer) where {IXT,N}
    iterlen = 1
    viewlen = 1
    for nx in 1:viewdims
        viewlen *= dest.shape[nx]
    end
    viewdims+=1
    for nx in viewdims:N
        iterlen *= dest.shape[nx]
    end
    
    di = MVector{N,IXT}(ntuple(Returns(1), Val(N))...)
    doffjump = ntuple(nx -> dest.strides[nx+1] - dest.shape[nx]*dest.strides[nx], Val(N-1))
    soffjump = ntuple(nx -> source.strides[nx+1] - source.shape[nx]*source.strides[nx], Val(N-1))
    di[viewdims] = 0
    dinar = dest.offset
    sinar = source.offset
    i = 0
    #@show viewlen, iterlen
    while i < iterlen
        #@show dinar
        i += 1
        
        #@show (dinar+1):(dinar+viewlen)
        @inbounds @view(dest.ar[(dinar+1):(dinar+viewlen)]) .= @inbounds @view(source.ar[(sinar+1):(sinar+viewlen)])
        
        dinar += dest.strides[viewdims]
        sinar += source.strides[viewdims]
        di[viewdims] += 1
        nx = viewdims
        while di[nx] > dest.shape[nx]
            di[nx+1] += 1
            di[nx] = 1
            dinar += @inbounds doffjump[nx]
            sinar += @inbounds soffjump[nx]
            nx += 1
        end
    end
    return nothing
end

function copy_ti!(dest::TensorIndex{IXT,N}, source::TensorIndex{IXT,N}) where {IXT,N}
    cd = min(continuous_dims(dest), continuous_dims(source))
    if cd == 0
        copy_ti_simple!(dest, source)
    elseif cd == N
        @inbounds @view(dest.ar[dest.offset:(dest.offset+l)]) .= @inbounds @view(source.ar[source.offset:(source.offset+l)])
    else
        copy_ti_views(dest, source, cd)
    end
    return nothing
end

function create_array(source::TensorIndex{IXT,N,T,AT}) where {IXT,N,T,AT}
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
                sinar += @inbounds source.strides[nx+1] - source.shape[nx]*source.strides[nx]
            end
        end
        @inbounds v[si...] = @inbounds source.ar[sinar]
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

function flatten(::Type{AT}, X::AbstractVector{<:AbstractArray{T,NP1}}; IXT=Int32) where {T,NP1,AT<:AbstractArray}
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
    return FlattenedTensorList{T, N, IXT, AT, Vector{FTLPositionInfo{IXT, N}}}(B, l, AT(flattened), positions)
end

function permute_features(ftl::FlattenedTensorList{T,N,IXT,AT,APT}, features::AbstractVector{IXT}) where {T,N,IXT,AT,APT}
    B = ftl.B
    l = sum(idx -> ftl.positions[idx][2], features)
    reordered = AT(undef, B*l)
    positions = APT(undef, length(features))
    position = 0
    for i in eachindex(features)
        positions[i] = ftl.positions[features[i]]
        positions[i].offset = position
        position += positions[i].len
    end
    newftl = FlattenedTensorList{T,N,IXT,AT,APT}(B, l, reordered, positions)
    for i in eachindex(features)
        copy!(feature_flat(newftl, i), feature_flat(ftl, features[i]))

        # @view(reshape(@view(reordered[:]), (l,B))[(position+1):(position+size), :]) .= 
        #     @view(reshape(@view(ftl.flattened[:]), (ftl.L, ftl.B))[(old_pos+1):(old_pos+size), :])
        
    end
    return newftl
end

function permute_features!(ftl::FlattenedTensorList, features::AbstractVector)
    ftl2 = permute_features(ftl, features)
    ftl.flattened = ftl2.flattened
    ftl.positions = ftl2.positions
    ftl.L = ftl2.L
    ftl.B = ftl2.B
    return ftl
end

# gets an element
@inline function Base.getindex(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer, bi::Integer, ixs::Vararg{Integer,N}) where {T,N,IXT,AT}
    acum = ftl.positions[fi].offset
    stride = ftl.positions[fi].strides
    fix = sum(NTuple{N,IXT}(ixs) .* (stride .- one(IXT))) + one(IXT)
    return ftl.flattened[ftl.L*(bi-1) + acum + fix]
end

# returns a reshape(view(AbstractArray)) representing a feature with all the samples
@inline function Base.getindex(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer) where {T,N,IXT,AT}
    acum = ftl.positions[fi].offset
    fl = ftl.positions[fi].len
    sizes = ftl.positions[fi].shape
    ffv = @view(reshape(@view(ftl.flattened[1:(ftl.L*ftl.B)]), (ftl.L, ftl.B))[(acum + 1):(acum + fl), :])
    return reshape(ffv, (sizes..., ftl.B))
end

# returns a reshape(view(AbstractArray)) representing a feature of the given sample
@inline function Base.getindex(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer, bi::Integer) where {T,N,IXT,AT}
    #acum, fl, sizes, _ = ftl.positions[fi]
    acum, = ftl.positions[fi].offset
    fl = ftl.positions[fi].len
    sizes = ftl.positions[fi].shape
    ffv = @view(reshape(@view(ftl.flattened[1:(ftl.L*ftl.B)]), (ftl.L, ftl.B))[(acum + 1):(acum + fl), bi])
    return reshape(ffv, sizes)
end

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