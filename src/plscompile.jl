
using DynamicExpressions.NodeModule: TensorNode
using DynamicExpressions.OperatorEnumModule: TensorOperatorEnum
using DynamicExpressions.FlattenedTensorListModule: FlattenedTensorList
using DynamicExpressions.OperatorEnumConstructionModule: @extend_operators, broadcast_binop, broadcast_unaop

c1 = TensorNode{Float64, 3}(; feature=1, constant=true)
c2 = TensorNode{Float64, 3}(; feature=2, constant=true)
c3 = TensorNode{Float64, 3}(; feature=3, constant=true)
c4 = TensorNode{Float64, 3}(; feature=4, constant=true)
c5 = TensorNode{Float64, 3}(; feature=5, constant=true)
x1 = TensorNode{Float64, 3}(; feature=1, constant=false)
x2 = TensorNode{Float64, 3}(; feature=2, constant=false)
x3 = TensorNode{Float64, 3}(; feature=3, constant=false)

function woaow(x::T) where {T<:Number}
    x^convert(T, 2) - x + one(T)
end

operators = TensorOperatorEnum(;
    binary_operators=[broadcast_binop(+), broadcast_binop(*)], unary_operators=[broadcast_unaop(woaow)]
)
@extend_operators(operators, on_type = Array{Float64,4})

trees = [
    matmult(transp(c4), cross(c1 + x1 * c2, x2 * c3)),
    matmult(matmult(c4, x3), (c1 + x1 * c2) * x2 * c3),
    matmult(c1, c2 + x2),
    matmult(c3, c4 + matmult(transp(matmult(c1, x1)), matmult(c2, c5 + transp(x2)))),
]

for tree in trees println(tree) end