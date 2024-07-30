
module FlattenedTensorListModule

const FTLPositionInfo = Tuple{IXT,IXT,NTuple{N,IXT},NTuple{N,IXT}} where {IXT,N}

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

function treat_as_flattened(buff::AT, sizes::AbstractVector{NTuple{N, IXT}}, max_B::Integer) where {T,IXT,N,AT<:AbstractVector{T}}
    positions = Vector{FTLPositionInfo{Int32,N}}(undef, length(sizes))

    acum = 0
    for fi in eachindex(sizes)
        positions[fi] = (acum, prod(sizes[fi]), sizes[fi], (1, cumprod(Base.front(sizes[fi]))...))
        acum += positions[fi][2]
    end
    l = positions[end][1] + positions[end][2]
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
    map!(Xi -> (0, div(length(Xi), B), Base.tail(size(Xi)), div.(Base.tail(strides(Xi)),B)), positions, X)
    acum = 0
    for fi in 1:f
        _, fl, size, stride = positions[fi] 
        positions[fi] = (acum, fl, size, stride)
        acum += positions[fi][2]
    end
    for bi in 1:B, fi in 1:f
        acum, fl, _, _ = positions[fi] 
        @view(flattened[(l*(bi-1)+acum+1):(l*(bi-1)+acum+fl)]) .= reshape(selectdim(X[fi], 1, bi), (fl,))
    end
    return FlattenedTensorList{T, N, IXT, AT, Vector{FTLPositionInfo{IXT, N}}}(B, l, AT(flattened), positions)
end

function permute_features(ftl::FlattenedTensorList{T,N,IXT,AT,APT}, features::AbstractVector{IXT}) where {T,N,IXT,AT,APT}
    B = ftl.B
    l = sum(idx -> ftl.positions[idx][2], features)
    reordered = AT(undef, B*l)
    positions = APT(undef, f)
    position = 0
    for i in eachindex(features)
        old_pos, size, sizes, strides = ftl.positions[features[i]]
        positions[i] = (position, size, sizes, strides)
        @view(reshape(@view(reordered[:]), (l,B))[(position+1):(position+size), :]) .= 
            @view(reshape(@view(ftl.flattened[:]), (ftl.L, ftl.B))[(old_pos+1):(old_pos+size), :])
        position += size
    end
    return FlattenedTensorList{T,N,IXT,AT,APT}(B, l, reordered, positions)
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
    acum, _, _, stride = ftl.positions[fi]
    fix = sum(NTuple{N,IXT}(ixs) .* (stride .- one(IXT))) + one(IXT)
    return ftl.flattened[ftl.L*(bi-1) + acum + fix]
end

# returns a reshape(view(AbstractArray)) representing a feature with all the samples
@inline function Base.getindex(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer) where {T,N,IXT,AT}
    acum, fl, sizes, _ = ftl.positions[fi]
    ffv = @view(reshape(@view(ftl.flattened[1:(ftl.L*ftl.B)]), (ftl.L, ftl.B))[(acum + 1):(acum + fl), :])
    return reshape(ffv, (sizes..., ftl.B))
end

# returns a reshape(view(AbstractArray)) representing a feature of the given sample
@inline function Base.getindex(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer, bi::Integer) where {T,N,IXT,AT}
    acum, fl, sizes, _ = ftl.positions[fi]
    ffv = @view(reshape(@view(ftl.flattened[1:(ftl.L*ftl.B)]), (ftl.L, ftl.B))[(acum + 1):(acum + fl), bi])
    return reshape(ffv, sizes)
end

function Base.display(ftl::FlattenedTensorList{T,N,IXT,AT}) where {T,N,IXT,AT}
    print("$(length(ftl.positions))-feature $(ftl.B)-sample $(ftl.L)-samplelength FlattenedTensorList{$(T),$(N)} of sizes:\n")
    for i in eachindex(ftl.positions)
        print("   $(ftl.positions[i][3]) at pos $(ftl.positions[i][1])\n")
    end
    # println("with values:")
    # for i in eachindex(ftl.positions)
    #     print("   ", ftl[i, 1], "\n")
    # end
end

end