
using DynamicExpressions.NodeModule: TensorNode
using DynamicExpressions.NodeUtilsModule: recalculate_node_values!
using DynamicExpressions.OperatorEnumModule
using DynamicExpressions.OperatorEnumModule: TensorOperator, TensorOperatorEnum
using DynamicExpressions.FlattenedTensorListModule: FlattenedTensorList, treat_as_flattened, flatten
using DynamicExpressions.OperatorEnumConstructionModule
using DynamicExpressions.OperatorEnumConstructionModule: broadcast_binop, broadcast_unaop, @extend_operators
using DynamicExpressions.ShapeInferenceModule
using DynamicExpressions.ShapeInferenceModule: @make_constraint, CombinedConstraints, Constraint, shape_inference
using DynamicExpressions.EvaluateTensorsModule: eval_diff_tree_array_cpu, eval_tree_array_cpu

c1 = TensorNode{Float32, 3}(; feature=1, constant=true)
c2 = TensorNode{Float32, 3}(; feature=2, constant=true)
c3 = TensorNode{Float32, 3}(; feature=3, constant=true)
c4 = TensorNode{Float32, 3}(; feature=4, constant=true)
c5 = TensorNode{Float32, 3}(; feature=5, constant=true)
x1 = TensorNode{Float32, 3}(; feature=1, constant=false)
x2 = TensorNode{Float32, 3}(; feature=2, constant=false)
x3 = TensorNode{Float32, 3}(; feature=3, constant=false)
x4 = TensorNode{Float32, 3}(; feature=3, constant=false)

loss(x, y) = (x-y)^2
woaw(x) = x^2-5*x+9

op_loss = TensorOperator(;
    symbol_name = :loss,
    op! = function(res::AbstractArray{T,N}, l, r) where {T,N}
        res[1] = sum((l.-r).^2)
    end,
    gradient! = function(res, l, dl, r)
        @. dl = 2*(l-r)
    end,
    push_constraints! = function(cs, (resoff, loff, roff), ::Val{N}) where {N}
        for nx in 1:N
            push!(cs, @make_constraint((resoff+nx, loff+nx, roff+nx), (1, n, n)))
        end
    end,
    complexity = (sl, sr) -> prod(sl)
)

op_mm = TensorOperator(;
    symbol_name = :mm,
    op! = function(res::AbstractArray{T,N}, l, r) where {T,N}
        # @show size(res)
        # @show size(l)
        # @show size(r)
        for i in axes(l, 1), j in axes(r, 2)
            selectdim(selectdim(res, 1, i), 1, j) .= 0
            for k in axes(l, 2)
                # @show i, j, k
                # @show selectdim(selectdim(res, 1, i), 1, j)
                # @show selectdim(selectdim(l, 1, i), 1, k)
                # @show selectdim(selectdim(r, 1, k), 1, j)
                selectdim(selectdim(res, 1, i), 1, j) .+= 
                    selectdim(selectdim(l, 1, i), 1, k) .* 
                    selectdim(selectdim(r, 1, k), 1, j)
            end
        end
    end,
    gradient! = function(res, dres, l, dl, r, dr, ::Val{comp}) where {comp}
        if comp & 0b01 != 0 # right
            # dr = mm(T(l), dres)
            for i in axes(dr, 1), j in axes(dr, 2)
                selectdim(selectdim(dr, 1, i), 1, j) .= 0
                for k in axes(dres, 1)
                    selectdim(selectdim(dr, 1, i), 1, j) .+= 
                        selectdim(selectdim(l, 1, k), 1, i) .* 
                        selectdim(selectdim(dres, 1, k), 1, j)
                end
            end
        end
        if comp & 0b10 != 0 # left
            for i in axes(dl, 1), j in axes(dl, 2)
                selectdim(selectdim(dl, 1, i), 1, j)
                for k in axes(dres, 2)
                    selectdim(selectdim(dl, 1, i), 1, j) .+= 
                        selectdim(selectdim(dres, 1, i), 1, k) .* 
                        selectdim(selectdim(r, 1, j), 1, k)
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
    op! = function(res::AbstractArray{T,N}, l, r) where {T,N}
        res .= rand(size(res)...)
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
    op! = function(res::AbstractArray{T,N}, l) where {T,N}
        permutedims!(res, l, ntuple(i -> i<3 ? 3-i : i, Val(N)))
    end,
    gradient! = function(res, dres, l, dl)
        permutedims!(dl, dres, ntuple(i -> i<3 ? 3-i : i, Val(N)))
    end,
    push_constraints! = function(cs, (resoff, loff), ::Val{N}) where {N}
        push!(cs, @make_constraint((resoff+1, resoff+2, loff+1, loff+2), (n, m, m, n)))
        for nx in 3:N
            push!(cs, @make_constraint((resoff+nx, loff+nx), (n, n)))
        end
    end,
    complexity = (sl) -> prod(sl)
)

operators = TensorOperatorEnum(;
    binary_operators=[broadcast_binop(+), broadcast_binop(*), broadcast_binop(-), op_mm, op_conv, op_loss], 
    unary_operators=[broadcast_unaop(-), broadcast_unaop(woaw), op_T]
)
@extend_operators operators

trees = [
    mm(T(c4), conv(c1 + woaw(x1) * c2, x2 * c3)),
    mm(mm(c4, x3), (c1 + x1 * c2) * x2 * c3),
    mm(c1, c2 + x2),
    mm(c3, c4 + mm(T(mm(c1, x1)), mm(c2, c5 + T(x2)))),
    mm(c1, x1) + x2
]

# for tree in trees println(tree) end


# ----------------------
# SHAPE INFERENCE EXAMPLE
# ----------------------

cb = CombinedConstraints(44, 5)
cs = Constraint[]
# a1 = a2 is equivalent to:
push!(cs, Constraint(
    Int32[1, 2],
    begin
        out = Array{Int32, 3}(undef, 1, 2, 2)
        out[1, 1, :] .= (0, 1)
        out[1, 2, :] .= (0, 1)
        out
    end
))
# a3 = 5 is equivalent to:
push!(cs, Constraint(
    Int32[3],
    Int32[5;;;]
))
# (a4,a5,a6) = {(n,1,n)} u {(n,n,n)} u {(n,n,1)} is:
push!(cs, Constraint(
    Int32[4, 5, 6],
    begin
        out = Array{Int32, 3}(undef, 3, 3, 2)
        out[1, 1, :] .= (0, 1)
        out[1, 2, :] .= (1, 0)
        out[1, 3, :] .= (0, 1)
        out[2, 1, :] .= (0, 1)
        out[2, 2, :] .= (0, 1)
        out[2, 3, :] .= (0, 1)
        out[3, 1, :] .= (0, 1)
        out[3, 2, :] .= (0, 1)
        out[3, 3, :] .= (1, 0)
        out
    end
))
# a9 = a7+a8 is:
push!(cs, Constraint(
    Int32[7, 8, 9],
    begin
        out = Array{Int32, 3}(undef, 1, 3, 3)
        out[1, 1, :] .= (0, 1, 0)
        out[1, 2, :] .= (0, 0, 1)
        out[1, 3, :] .= (0, 1, 1)
        out
    end
))

cb.values[3, 3] = 3
cb.values[4, 5] = 6
cb.values[3, 5] = 3
cb.values[1, 5] = 2
cb.values[1, 1] = 9
cb.values[1, 2] = 9
cb.values[1, 3] = 1
# print(cb)
# print(cs)
# cX = treat_as_flattened(buffer, [(3, 3, 1), (1, 3, 1), (3, 1, 1), (1, 1, 1)], 2)
# for i in 1:4
#     println("DOING TREE ", i)
#     try
#        @time shape_inference(trees[i], operators, cX)
#     catch c
#         println(c)
#         println("TREE ", i, " FAILED")
#     end
# end
# println(cX2)
# println(cC)
# println(methods(eval_diff_tree_array_cpu))
# println(typeof(cX2))
# println(typeof(cC))
# println(typeof(trees[5]))
# println(typeof(buffer))


buffer = rand(Float32, 120)

# the inputs, with the labels at the end
# cX = [x1|x2|x3|y]
cX = flatten(Vector{Float32}, [rand(Float32, 10, 4, 1, 1), rand(Float32, 10, 4, 1, 1), rand(Float32, 10, 4, 1, 1), rand(Float32, 10, 4, 1, 1)])

# the result of the operation will be computed here (if you don't compute the derivative)
results = flatten(Vector{Float32}, [rand(Float32, 10, 4, 1, 1)])

# the constants that are used in the expression
# the first is the sample is the actual value and the second is the to-be-computed derivative
cC = flatten(Vector{Float32}, [rand(Float32, 2, 4, 4, 1), rand(Float32, 2, 4, 1, 1)])

# infers the shapes and puts them into the tree
# later this will have a specific shape generator
shape_inference(trees[5], operators, cX)
recalculate_node_values!(trees[5], cX)

reducer_op = op_loss

# options:

#                  the structured expression  inputs   where to store   working
#                                                      the results      memory
eval_tree_array_cpu(trees[5], cC, operators,  cX,      results,         buffer)

# the reducer_op is applied for every sample to the result and  the last element of cX (the labels) 
# to obtain a scalar, which is summed over the batch size to return a scalar 
eval_tree_array_cpu(trees[5], cC, operators,  cX,      reducer_op,       buffer)

# the reducer_op is applied for every sample to obtain a scalar, which is summed over the batch size to return a scalar 
# the derivatives with respect to the final scalar are also written into the second sample in the constants ftl
eval_diff_tree_array_cpu(trees[5], cC, operators,  cX, reducer_op,       buffer)

