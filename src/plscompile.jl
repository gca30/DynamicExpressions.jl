
using DynamicExpressions.NodeModule: TensorNode
using DynamicExpressions.OperatorEnumModule
using DynamicExpressions.OperatorEnumModule: TensorOperator, TensorOperatorEnum
using DynamicExpressions.FlattenedTensorListModule: FlattenedTensorList, treat_as_flattened
using DynamicExpressions.OperatorEnumConstructionModule
using DynamicExpressions.OperatorEnumConstructionModule: broadcast_binop, broadcast_unaop, @extend_operators
using DynamicExpressions.ShapeInferenceModule
using DynamicExpressions.ShapeInferenceModule: @make_constraint, CombinedConstraints, Constraint, shape_inference

c1 = TensorNode{Float64, 3}(; feature=1, constant=true)
c2 = TensorNode{Float64, 3}(; feature=2, constant=true)
c3 = TensorNode{Float64, 3}(; feature=3, constant=true)
c4 = TensorNode{Float64, 3}(; feature=4, constant=true)
c5 = TensorNode{Float64, 3}(; feature=5, constant=true)
x1 = TensorNode{Float64, 3}(; feature=1, constant=false)
x2 = TensorNode{Float64, 3}(; feature=2, constant=false)
x3 = TensorNode{Float64, 3}(; feature=3, constant=false)

function woaw(x::T) where {T<:Number}
    x^convert(T, 2) - x + one(T)
end

op_mm = TensorOperator(;
    symbol_name = :mm,
    op! = function(res::AbstractArray{T,N}, l, r) where {T,N}
        @assert N >= 2
        @assert size(x)[3:end] == size(y)[3:end]
        @assert size(x, 2) == size(y, 1)
        res .= rand(size(l, 1), size(r, 2), size(l)[3:end]...)
    end,
    gradient! = function(res, dres, l, dl, r, dr, ::Val{comp}) where {comp}
        # TODO: implement this and op!
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
    binary_operators=[broadcast_binop(+), broadcast_binop(*), broadcast_binop(-), op_mm, op_conv], 
    unary_operators=[broadcast_unaop(-), broadcast_unaop(woaw), op_T]
)
@extend_operators operators

trees = [
    mm(T(c4), conv(c1 + woaw(x1) * c2, x2 * c3)),
    mm(mm(c4, x3), (c1 + x1 * c2) * x2 * c3),
    mm(c1, c2 + x2),
    mm(c3, c4 + mm(T(mm(c1, x1)), mm(c2, c5 + T(x2)))),
]

for tree in trees println(tree) end


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


buffer = Vector{Float64}(undef, 32)
cX = treat_as_flattened(buffer, [(3, 3, 1), (1, 3, 1), (3, 1, 1), (1, 1, 1)], 2)

for i in 1:4
    print("\n\n\nDOING TREE ", i, "\n\n")
    try
  #      shape_inference(trees[i], operators, cX)
    catch c
        println(stderr, c)
        println("TREE ", i, " FAILED\n\n")
    end
end
