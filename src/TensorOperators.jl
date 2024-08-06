
module TensorOperatorsModule

using ..ShapeInferenceModule: push_constraints_broadcast, @make_constraint, Constraint
using ..FlattenedTensorListModule: mapk_ti!, zero_ti!, addto_ti!, selectdim_ti, TensorIndex
using ..OperatorEnumModule: TensorOperator

# TODO: get gradient from somewhere else
gradient(::typeof(+), l) = 1
gradient(::typeof(-), l) = -1
gradientl(::typeof(+), l, r) = 1
gradientr(::typeof(+), l, r) = 1
gradientl(::typeof(-), l, r) = 1
gradientr(::typeof(-), l, r) = -1
gradientl(::typeof(*), l, r) = r
gradientr(::typeof(*), l, r) = l

# now the operators accept TensorIndex, with the last dimension being the batch size
broadcast_unaop(op::Fnum; op_complexity=1, symbol::Union{Symbol, Nothing}=nothing) where {Fnum} = TensorOperator(;
    symbol_name = symbol === nothing ? Symbol(op) : symbol,
    op! = (res, l) -> mapk_ti!((rest, lt) -> op(lt), res, l),
    # (l, res) -> (@. res = op(l)),
    gradient! = (res, ∂res, l, ∂l) -> mapk_ti!((dlt, lt, rest) -> gradient(op, lt)*rest, ∂l, l, ∂res),
    push_constraints! = push_constraints_broadcast,
    complexity = sl -> length(sl) * op_complexity
)

broadcast_binop(op::Fnum ; op_complexity=1, symbol::Union{Symbol, Nothing}=nothing) where {Fnum} = TensorOperator(
    symbol_name = symbol === nothing ? Symbol(op) : symbol,
    op! = (res, l, r) -> mapk_ti!((rest, lt, rt) -> op(lt, rt), res, l, r),
    # (l, r, res) -> (@. res = op(l, r)),
    gradient! = function(res, ∂res, l, ∂l, r, ∂r, ::Val{comp}) where {comp}
        if comp & 0b10 == 0b10
            mapk_ti!((dlt, lt, rt, drest) -> gradientl(op, lt, rt)*drest, ∂l, l, r, ∂res)
        end
        if comp & 0b01 == 0b01
            mapk_ti!((drt, lt, rt, drest) -> gradientr(op, lt, rt)*drest, ∂r, l, r, ∂res)
        end
    end,
    push_constraints! = push_constraints_broadcast,
    complexity = (sl, sr) -> 
        prod(ntuple(i -> max(sl[i], sr[i]), Val(length(sl)))) * op_complexity
)

op_mse_loss = TensorOperator(;
    symbol_name = :loss,
    op! = function(res::TensorIndex{IXT,NP1,T}, l, r) where {IXT,NP1,T}
        #for bx in 1:res.shape[NP1]
        #r2 = selectdim_ti(res, NP1, bx)
        zero_ti!(res)
        addto_ti!((lt, rt) -> (lt-rt)^2, res, l, r)
        #end
    end,
    gradient! = function(res, l, dl, r)
        mapk_ti!((_, lt, rt) -> 2*(lt-rt), dl, l, r)
    end,
    push_constraints! = function(cs, (resoff, loff, roff), ::Val{N}) where {N}
        for nx in 1:N
            push!(cs, @make_constraint((resoff+nx, loff+nx, roff+nx), (1, n, n)))
        end
    end,
    complexity = (sl, sr) -> prod(sl)
)

@inline seldims(x, i, j) = selectdim_ti(selectdim_ti(x, 1, i), 2, j) #x[i, j, ntuple(Returns(:), Val(N-2))...]

op_mm = TensorOperator(;
    symbol_name = :mm,
    op! = function(res::TensorIndex{IXT,NP1,T}, l, r) where {IXT,NP1,T}
        # @show size(res)
        # @show size(l)
        # @show size(r)
        zero_ti!(res)
        for i in axes(l, 1), j in axes(r, 2)
            #@show i, j
            for k in axes(l, 2)
             #   @show i, j, k
                # @show selectdim(selectdim(res, 1, i), 1, j)
                # @show selectdim(selectdim(l, 1, i), 1, k)
                # @show selectdim(selectdim(r, 1, k), 1, j)
                # @show seldims(res, i, j).shape
                # @show seldims(l, i, k).shape
                # @show seldims(r, k, j)
                #@show "before", materialize_ti(selectdim_ti(res, NP1, 1:10))
                addto_ti!((l, r) -> l*r, seldims(res, i, j), seldims(l, i, k), seldims(r, k, j))
                #@show "after", materialize_ti(selectdim_ti(res, NP1, 1:10))
            end
        end
        #@show materialize_ti(selectdim_ti(res, NP1, 1:100))
    end,
    gradient! = function(res::TensorIndex{IXT,N,T}, dres, l, dl, r, dr, ::Val{comp}) where {comp,IXT,N,T}
        if comp & 0b01 != 0 # right
            # DR = L^T DRES
            zero_ti!(dr)
            for i in axes(dr, 1), j in axes(dr, 2)
                for k in axes(dres, 1)
                    addto_ti!((l, r) -> l*r, seldims(dr, i, j), seldims(l, k, i), seldims(dres, k, j))
                end
            end
        end
        if comp & 0b10 != 0 # left
            # DL = DRES R^T
            zero_ti!(dl)
            for i in axes(dl, 1), j in axes(dl, 2)
                for k in axes(dres, 2)
                    addto_ti!((l, r) -> l*r, seldims(dl, i, j), seldims(dres, i, k), seldims(r, j, k))
                end
            end
        end
    end,
    push_constraints! = function(cs, (resoff, loff, roff), ::Val{N}) where {N}
        push!(cs, @make_constraint((resoff+1, resoff+2, loff+1, loff+2, roff+1, roff+2), (n, p, n, m, m, p)))
        for nx in 3:N
            push!(cs, @make_constraint((resoff+nx, loff+nx, roff+nx), (n, n, n)))
        end
    end,
    complexity = (sl, sr) -> prod(sl[3:end]) * sl[1] * sr[2]
)

op_conv = TensorOperator(;
    symbol_name = :conv,
    op! = function(res, l, r)
        # res .= rand(size(res)...)
    end,
    gradient! = function(res, dres, l, dl, r, dr, ::Val{comp}) where {comp}
        # TODO: implement this and op!
    end,
    push_constraints! = function(cs, (resoff, loff, roff), ::Val{N}) where {N}
        for nx in 1:N
            # for a given n or m, only one of these cases will be correct
            push!(cs, @make_constraint((resoff+nx, loff+nx, roff+nx), (n+1-m, n, m), (m+1-n, n, m)))
        end
    end,
    complexity = (sl, sr) -> prod(abs.(sl.-sr).+1) * prod(min.(sl, sr))
)

op_T = TensorOperator(;
    symbol_name = :T,
    op! = function(res, l)
        #permutedims!(res, l, ntuple(i -> i<3 ? 3-i : i, Val(N)))
        # TODO: implement this
    end,
    gradient! = function(res, dres, l, dl)
        #permutedims!(dl, dres, ntuple(i -> i<3 ? 3-i : i, Val(N)))
    end,
    push_constraints! = function(cs, (resoff, loff), ::Val{N}) where {N}
        push!(cs, @make_constraint((resoff+1, resoff+2, loff+1, loff+2), (n, m, m, n)))
        for nx in 3:N
            push!(cs, @make_constraint((resoff+nx, loff+nx), (n, n)))
        end
    end,
    complexity = (sl) -> prod(sl)
)

end