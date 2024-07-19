
using CUDA

const FTLPositionInfo = Tuple{IXT,IXT,NTuple{N,IXT},NTuple{N,IXT}} where {IXT,N}

# The FlattenedTensorList stores a flattened representation of a list of list of tensors.
# The first axis is the batch axis. The second axis is the feature axis.
# Different features can have diferent shapes, but along the batch axes, the shapes are the same.
# The flattened representation also stores the batch axis last, so it shouldn't have problems with cache.
struct FlattenedTensorList{T,N,IXT,AT<:AbstractVector{T}}
    B::IXT # number of samples
    L::IXT # total length in Ts of each sample
    flattened::AT # A B*L array of Ts
    positions::Vector{FTLPositionInfo{IXT,N}}
    # for each feature:
    #   the index into the flattened array at which it starts at
    #   the length of the feature
    #   N sizes representing the size of the tensor
    #   N strides
end

function treat_as_flattened(buff::AT, sizes::Vector{NTuple{N, IXT}}) where {T,IXT,N,AT<:AbstractVector{T}}
    positions = Vector{FTLPositionInfo{Int32,N}}(undef, f)
    map!(csize -> (0, prod(csize), csize, (1, cumprod(Base.front(csize))...)), positions, sizes)
    l = positions[end][1] + positions[end][2]
    b = div(length(buff), l)
    return FlattenedTensorList{T,N,IXT,AT}(b, l, buff, positions)
end

function flatten_cu(X::AbstractVector{<:AbstractArray{T,NP1}}) where {T,NP1}
    N = NP1-1
    B = size(X[1], 1)
    l = sum(Xi -> div(length(Xi), B), X)
    f = length(X)
    flattened = Array{T,1}(undef, B*l)
    positions = Vector{FTLPositionInfo{Int32,N}}(undef, f)
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
    return FlattenedTensorList{T, N, Int32, CuArray{T, 1, CUDA.Mem.DeviceBuffer}}(B, l, cu(flattened), positions)
end

# gets an element
@inline function Base.getindex(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer, bi::Integer, ixs::Vararg{Integer,N}) where {T,N,IXT,AT}
    acum, _, _, stride = ftl.positions[fi]
    fix = sum(NTuple{N,IXT}(ixs) .* (stride .- one(IXT))) + one(IXT)
    return ftl.flattened[ftl.L*(bi-1) + acum + fix]
end

# returns a reshape(view(CuArray)) representing a feature with all the samples
@inline function Base.getindex(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer) where {T,N,IXT,AT}
    acum, fl, sizes, _ = ftl.positions[fi]
    ffv = @view(reshape(ftl.flattened, (ftl.L, ftl.B))[(acum + 1):(acum + fl), :])
    return reshape(ffv, (sizes..., ftl.B))
end

# returns a reshape(view(CuArray)) representing a feature of the given sample
@inline function Base.getindex(ftl::FlattenedTensorList{T,N,IXT,AT}, fi::Integer, bi::Integer) where {T,N,IXT,AT}
    acum, fl, sizes, _ = ftl.positions[fi]
    ffv = @view(reshape(ftl.flattened, (ftl.L, ftl.B))[(acum + 1):(acum + fl), bi])
    return reshape(ffv, (sizes...))
end

# For evaluation on the GPU, we must:
#   1. Before any evaluation, we flatten the inputs, [X|y] into a LxB array.
#       -> An example of the array flattening procedure is the one in this file.
#   2. We create a temporary variables array (let's say of size M = 1.5 GB).
#       -> These two arrays will never see the CPU again.
# Now given the expression:
#   1. Create a flattened constants array and a flattened constants' derivatives array.
#   2. Calculate what temporary variables we need, what temporary derivatives we need, etc. All these will have a total length l.
#   3. Find the evaluation batch size b = min(B, M/l). We start treating the memory buffer lxb flattened array.
#   4. Construct the shape buffer, which holds all the shapes and positions of the arrays in the memory buffer.
#   5. Construct a command buffer, that tells the GPU kernel what operations to execute at the same time, how many threads per layer, etc.
#   6. Construct the gradient operations into the command buffer.
#   7. Execute the kernel, which interprets the command buffer, given shapes in the shape buffer and finds the right one.
#   8. Must repeat the kernel div(B,b) times. Each time we add the constants' gradient into the constants gradients buffer.
#
# We must repeat this B/b times because we don't know if the GPU can store every temporary variable. 
# For example, on my laptop, only about 15 MNISTs fit in my VRAM. For mildly complicated expressions, we get more than that, so we must batch.
#

# The command buffer will have the following format:
# Number of layers
# for every layer of computation: (position in the command buffer, length in the command buffer)
# Number of operations
# for every operation: (degree, operator, l_index, l_source, r_index, r_source, threads)
#   with index = the feature number / the constant number / the temp buffer number 
#        source = inputs / constants / temp buffer
#        threads = number of threads to allocate for the operation
#   the operator will have a macro that takes a number from 1:threads and returns what to do in that thread
#   a macro is necesary so that the gpu kernel compiles instead of calling functions (which might not work on the gpu)

