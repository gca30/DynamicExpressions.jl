
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

# This might be faster than using views, which somehow cause copying
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

function mapk_ti!(op::F, dest::TensorIndex{IXT,N}, sources::Vararg{TensorIndex{IXT,N}, K}) where {F,IXT,N,K}

    itershape = ntuple(nx -> max(dest.shape[nx], maximum(source -> source.shape[nx], sources; init=0)), Val(N))
    iterlen = prod(itershape)
    ii = MVector{N,IXT}(ntuple(Returns(1), Val(N))...)
    sinar = MVector{K,IXT}(ntuple(kx -> sources[kx].offset+1, Val(K)))    
    # @show itershape

    for i in 1:iterlen

        nx = 1
        while ii[nx] > itershape[nx]
            ii[nx+1] += 1
            ii[nx] = 1
            nx+=1
        end
        dinar = index_clamp(dest, ii...)
        for kx in 1:K
            sinar[kx] = index_clamp(sources[kx], ii...)
        end
        # @show dinar
        # @show sinar
        # @show ii
        ii[1] += 1

        dest.ar[dinar] = op(
            (dest.ar[dinar]),
            ntuple(kx -> (sources[kx].ar[sinar[kx]]), Val(K))...
        )
        
    end
    return nothing

end

function _copy_ti_simple!(op::F, dest::TensorIndex{IXT,N}, source::TensorIndex{IXT,N}) where {IXT,N,F}
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
            dinar +=  doffjump[nx]
            sinar +=  soffjump[nx]
            nx += 1
        end
         dest.ar[dinar] = op( source.ar[sinar])
    end
    return nothing
end

function _copy_ti_views(op::F, dest::TensorIndex{IXT,N}, source::TensorIndex{IXT,N}, viewdims::Integer) where {IXT,N,F}
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
    while i < iterlen
        i += 1
        for j in 1:viewlen
             dest.ar[dinar+j] = op( source.ar[sinar+j])
        end
        
        dinar += dest.strides[viewdims]
        sinar += source.strides[viewdims]
        di[viewdims] += 1
        nx = viewdims
        while di[nx] > dest.shape[nx]
            di[nx+1] += 1
            di[nx] = 1
            dinar +=  doffjump[nx]
            sinar +=  soffjump[nx]
            nx += 1
        end
    end
    return nothing
end

# using a 100_000x5x5x5 array: 
#                without op  with op
# new copy        ~14.5 ms    ~16.0ms    copy_ti!(feature(ftl, 1), feature(ftl, 2))
# new copy flat   ~14.4 ms    ~14.6ms    copy_ti!(feature_flat(ftl, 1), feature_flat(ftl, 2))
# new copy simple ~14ms      ~14ms       copy_ti_simple!(op, feature_flat(ftl, 1), feature_flat(ftl, 2))
# raw vectors     ~5.8 ms     ~9.6ms     x1 .= x2
# 1-dim views     ~9.4 ms     ~9.4ms     @view(x1[:]) .= @view(x1[:]
# N-dim views     ~14.5 ms    ~7.8ms     @view(x1[1:B,:,:,:]) .= @view(x1[1:B,:,:,:]
# before          ~1.7s       ~660ms     ftl[1] .= ftl[2] (due to copying somewhere)
function copy_ti!(op::F, dest::TensorIndex{IXT,N}, source::TensorIndex{IXT,N}) where {IXT, N, F}
    cd = min(continuous_dims(dest), continuous_dims(source))
    if cd == 0
        _copy_ti_simple!(op, dest, source)
    elseif cd == N
         @view(dest.ar[dest.offset:(dest.offset+length(dest))]) .= op.( @view(source.ar[source.offset:(source.offset+length(dest))]))
    else
        _copy_ti_views(op, dest, source, cd)
    end
    return nothing
end

function copy_ti!(dest::TensorIndex{IXT,N}, source::TensorIndex{IXT,N}) where {IXT,N}
    cd = min(continuous_dims(dest), continuous_dims(source))
    if cd == 0
        _copy_ti_simple!(identity, dest, source)
    elseif cd == N
         @view(dest.ar[dest.offset:(dest.offset+l)]) .=  @view(source.ar[source.offset:(source.offset+l)])
    else
        _copy_ti_views(identity, dest, source, cd)
    end
    return nothing
end

# Benchmarks for 100_000:
# first version: 15.8ms
# second version: 15.0mms
# raw arrays: 2.5ms
# views:      2.5ms
# before:     220ms
# still a 15x improvement, but not compared to the 80x improvement for raw arrays
function _map2b_ti!(op::F, dest::TensorIndex{IXT,N}, lterm::TensorIndex{IXT,N}, rterm::TensorIndex{IXT,N}) where {IXT,N,F}

    itershape = ntuple(nx -> max(dest.shape[nx], lterm.shape[nx], rterm.shape[nx]), Val(N))
    len = prod(itershape)

    ii = MVector{N,IXT}(ntuple(Returns(1), Val(N))...)
    doffjump = ntuple(nx -> if nx == 1
        itershape[1] == dest.shape[1] ? dest.strides[1] : 0 # iteration stride
    elseif itershape[nx-1] != dest.shape[nx-1]
        dest.strides[nx-1] # if the previous overflown axis was broadcasted, you need to advance to the next element
    elseif itershape[nx] == dest.shape[nx]
        dest.strides[nx] - dest.shape[nx-1]*dest.strides[nx-1] # from the overflown last element to the next first element
    else
        -dest.shape[nx-1]*dest.strides[nx-1] # if you overflow into a broadcasted axis, go back to 1
    end, Val(N))
    loffjump = ntuple(nx -> if nx == 1
        itershape[1] == lterm.shape[1] ? lterm.strides[1] : 0 # iteration stride
    elseif itershape[nx-1] != lterm.shape[nx-1]
        lterm.strides[nx-1] # if the previous overflown axis was broadcasted, you need to advance to the next element
    elseif itershape[nx] == lterm.shape[nx]
        lterm.strides[nx] - lterm.shape[nx-1]*lterm.strides[nx-1] # from the overflown last element to the next first element
    else
        -lterm.shape[nx-1]*lterm.strides[nx-1] # if you overflow into a broadcasted axis, go back to 1
    end, Val(N))

    roffjump = ntuple(nx -> if nx == 1
        itershape[1] == rterm.shape[1] ? rterm.strides[1] : 0
    elseif itershape[nx-1] != rterm.shape[nx-1]
        rterm.strides[nx-1]
    elseif itershape[nx] == rterm.shape[nx]
        rterm.strides[nx] - rterm.shape[nx-1]*rterm.strides[nx-1]
    else
        -rterm.shape[nx-1]*rterm.strides[nx-1]
    end, Val(N))
    
    dinar = dest.offset+1
    rinar = rterm.offset+1
    linar = lterm.offset+1
 
    i = 0
    while i < len
        i += 1

        nx = 1
        while ii[nx] > itershape[nx]
            ii[nx+1] += 1
            ii[nx] = 1
            dinar += doffjump[nx+1]
            rinar += roffjump[nx+1]
            linar += loffjump[nx+1]
            nx += 1
        end
        
         dest.ar[dinar] = op(( dest.ar[dinar]), ( lterm.ar[linar]), ( rterm.ar[rinar]))

        ii[1] += 1
        dinar += doffjump[1]
        linar += loffjump[1]
        rinar += roffjump[1]
        
    end
    return nothing
end

# for 100_000x10x10:
# _map2nbv_ti! 1ms
# raw arrays   9ms
# views        10ms
function _map2nbv_ti!(op::F, dest::TensorIndex{IXT,N}, lterm::TensorIndex{IXT,N}, rterm::TensorIndex{IXT,N}, viewdims::Integer) where {IXT,N,F}

    #cd = min(continuous_dims(dest), continuous_dims(rterm), continuous_dims(lterm))
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
    loffjump = ntuple(nx -> lterm.strides[nx+1] - lterm.shape[nx]*lterm.strides[nx], Val(N-1))
    roffjump = ntuple(nx -> rterm.strides[nx+1] - rterm.shape[nx]*rterm.strides[nx], Val(N-1))
    dinar = dest.offset
    linar = lterm.offset
    rinar = rterm.offset
    i = 0
    while i < iterlen
        i += 1
        nx = viewdims
        while di[nx] > dest.shape[nx]
            di[nx+1] += 1
            di[nx] = 1
            dinar +=  doffjump[nx]
            linar +=  loffjump[nx]
            rinar +=  roffjump[nx]
            nx += 1
        end

        for j in 1:viewlen
             dest.ar[dinar+j] = op(
                ( dest.ar[dinar+j]),
                ( lterm.ar[linar+j]),
                ( rterm.ar[rinar+j])
            )
        end
        
        dinar += dest.strides[viewdims]
        if dinar - dest.offset < 300
            @show di, dinar, linar, rinar
        end
        linar += lterm.strides[viewdims]
        rinar += rterm.strides[viewdims]
        di[viewdims] += 1
        
    end
    return nothing
end

function map2_ti!(op::F, dest::TensorIndex{IXT,N}, lterm::TensorIndex{IXT,N}, rterm::TensorIndex{IXT,N}) where {IXT,N,F}
    if all(ntuple(nx -> dest.shape[nx] == lterm.shape[nx] && dest.shape[nx] == rterm.shape[nx], Val(N)))
        cd = min(continuous_dims(dest), continuous_dims(lterm), continuous_dims(rterm))
        if cd == N
            A = @view(dest.ar[(dest.offset+1):(dest.offset+length(dest))])
            B = @view(lterm.ar[(lterm.offset+1):(lterm.offset+length(lterm))])
            C = @view(rterm.ar[(rterm.offset+1):(rterm.offset+length(rterm))])
            for i in 1:length(A)
                A[i] = op(A[i], B[i], C[i])
            end
            #@. A = op(B,C)
        else
            println("_map2nbv_ti!")
            _map2nbv_ti!(op, dest, lterm, rterm, cd)
        end
    else
        println("_map2b_ti!")
        _map2b_ti!(op, dest, lterm, rterm)
    end
end

function reducedim_ti!(op::F, reduce_op::Fr, dest::TensorIndex{IXT,N,T}, source::TensorIndex{IXT,N}) where {IXT,N,T,F,Fr}
    len = length(dest)
    si = MVector{N,IXT}(ntuple(Returns(1), Val(N))...)
    doffjump = ntuple(nx -> nx == 1 ? dest.strides[1] : dest.strides[nx] - dest.shape[nx-1]*dest.strides[nx-1], Val(N))
    soffjump = ntuple(nx -> nx == 1 ? source.strides[1] : source.strides[nx] - source.shape[nx-1]*source.strides[nx-1], Val(N))
    
    dinar = dest.offset+1
    sinar = source.offset+1
 
    i = 0
    while i < len
        i += 1

        nx = 1
        while si[nx] > source.shape[nx]
            si[nx+1] += 1
            si[nx] = 1
            sinar += soffjump[nx+1]
            nx += 1
        end
        
         dest.ar[dinar] = reduce_op(( dest.ar[dinar]), ( source.ar[sinar]))

        si[1] += 1
        dinar += doffjump[1]
        linar += loffjump[1]
        rinar += roffjump[1]
        
    end
end

function mapreduce2nb_ti!(op::F, reduce_op::Fr, startr::T, lterm::TensorIndex{IXT,N,T}, rterm::TensorIndex{IXT,N}) where {IXT,N,T,F,Fr}
    iterlen = length(lterm)

    di = MVector{N,IXT}(ntuple(Returns(1), Val(N))...)
    loffjump = ntuple(nx -> lterm.strides[nx+1] - lterm.shape[nx]*lterm.strides[nx], Val(N-1))
    roffjump = ntuple(nx -> rterm.strides[nx+1] - rterm.shape[nx]*rterm.strides[nx], Val(N-1))
    linar = lterm.offset+1
    rinar = rterm.offset+1
    i = 0

    while true
        i += 1
        startr = reduce_op(startr, op(lterm.ar[linar], rterm.ar[rinar]))
        
        linar += lterm.strides[1]
        rinar += rterm.strides[1]

        if i >= iterlen
            break
        end

        di[1] += 1
        nx = 1
        while di[nx] > lterm.shape[nx]
            di[nx+1] += 1
            di[nx] = 1
            linar +=  loffjump[nx]
            rinar +=  roffjump[nx]
            nx += 1
        end
    end

    return startr
end

function mapnb_ti!(op::F, dest::TensorIndex{IXT,N}, terms::Vararg{TensorIndex{IXT,N},M}) where {IXT,N,M,F}
    iterlen = length(dest)

    di = MVector{N,IXT}(ntuple(Returns(1), Val(N))...)
    doffjump = NTuple{N-1, IXT}(ntuple(nx -> dest.strides[nx+1] - dest.shape[nx]*dest.strides[nx], Val(N-1)))
    toffjump = NTuple{M, NTuple{N-1, IXT}}(ntuple(mx -> ntuple(nx -> terms[mx].strides[nx+1] - terms[mx].shape[nx]*terms[mx].strides[nx], Val(N-1)), Val(M)))
    # @show doffjump
    # @show dest.strides
    # @show map(t -> t.strides, terms)
    # @show toffjump
    tinar = MVector{M,IXT}(ntuple(mx -> terms[mx].offset+1, Val(M))...)
    dinar = IXT(dest.offset+1)
    i = 0

    while true
        i += 1
         dest.ar[dinar] = op(( dest.ar[dinar]), ntuple(mx -> ( terms[mx].ar[ tinar[mx]]), Val(M))...)
        
        if i >= iterlen
            break
        end

        for mx in 1:M
            tinar[mx] +=  terms[mx].strides[1]
        end
        dinar += dest.strides[1]
        di[1] += 1
        nx = one(IXT)
        while  di[nx] > dest.shape[nx]
             di[nx+1] += 1
             di[nx] = 1
            for mx in 1:M
                tinar[mx] +=  toffjump[mx][nx]
            end
            dinar +=  doffjump[nx]
            nx += 1
        end
        # @show di

        @assert dinar == index(dest, di...)
        for mx in 1:M
          #  @show mx
            if tinar[mx] != index(terms[mx], di...)
                @show tinar[mx]
                @show di
                @show dest.shape
                @show toffjump[mx]
                @show mx
                @show terms[mx].shape
                @show terms[mx].offset
                @show terms[mx].strides
                @show index(terms[mx], di...)
                error("MISMATCH")
            end
        end
    end
    return nothing
end

# zeroing out a 3000 vector
# x .= 0   -> 80ns
# x .= Returns(0).(x) -> 110 ns
# mapself_ti!(Returns(0), f) -> 33ns (how is this faster?)
function mapself_ti!(op::F, dest::TensorIndex{IXT,N}) where {F,IXT,N}
    copy_ti!(op, dest, dest) # TODO: maybe if we rewrite this it will be faster
end

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
        copy_ti!(feature_flat(newftl, i), feature_flat(ftl, features[i]))
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