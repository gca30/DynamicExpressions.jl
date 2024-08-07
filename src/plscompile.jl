
using DynamicExpressions.NodeModule: TensorNode
using DynamicExpressions.TensorNodeUtilsModule: recalculate_node_values!, reshape_constants
using DynamicExpressions.OperatorEnumModule
using DynamicExpressions.OperatorEnumModule: TensorOperator, TensorOperatorEnum
using DynamicExpressions.FlattenedTensorListModule: FlattenedTensorList, treat_as_flattened, flatten
using DynamicExpressions.OperatorEnumConstructionModule: @extend_operators
using DynamicExpressions.TensorOperatorsModule: broadcast_binop, broadcast_unaop, op_mm, op_conv, op_mse_loss, op_T
using DynamicExpressions.ShapeInferenceModule: reshape_inference
using DynamicExpressions.EvaluateTensorsModule: eval_diff_tree_array_cpu, eval_tree_array_cpu, eval_diff_tree_array_gpu
using DynamicExpressions.StringsModule: string_debug_tree
using DynamicExpressions.TensorExpressionModule: make_tensor_expression, set_constants

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
    mm(c1, x1) + x2
]

# for tree in trees println(tree) end
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
# println(constants)
# println(methods(eval_diff_tree_array_cpu))
# println(typeof(cX2))
# println(typeof(constants))
# println(typeof(trees[5]))
# println(typeof(buffer))


buffer = Vector{Float32}(undef, 1000_000)

# the inputs, with the labels at the end
# cX = [x1|x2|x3|y]
BBB = 3_000
println("Making cX")
cX = flatten(Vector{Float32}, [rand(Float32, BBB, 4, 1, 1), rand(Float32, BBB, 4, 1, 1), rand(Float32, BBB, 4, 1, 1), rand(Float32, BBB, 4, 1, 1)])

constants = flatten(Vector{Float32}, [rand(Float32, 2, 4, 1, 1), rand(Float32, 2, 4, 1, 1)])
te = make_tensor_expression(trees[5], constants, operators)
reshape_inference(te, cX)


reducer_op = op_mse_loss

eval_diff_tree_array_cpu(trees[5], constants, operators,  cX, reducer_op,       buffer)

BBB = 11


function redo_thing()
    # global BBB
    global cX
    # global results
    global constants
    # global buffer

    #println("OK, now we start")
    buffer .= 666
    cX = flatten(Vector{Float32}, [rand(Float32, BBB, 4, 1, 1), rand(Float32, BBB, 4, 1, 1), rand(Float32, BBB, 4, 1, 1), rand(Float32, BBB, 4, 1, 1)])

    # the result of the operation will be computed here (if you don't compute the derivative)
    # results = flatten(Vector{Float32}, [rand(Float32, BBB, 4, 1, 1)])

    # the constants that are used in the expression
    # the first is the sample is the actual value and the second is the to-be-computed derivative

    # infers the shapes and puts them into the tree
    # later this will have a specific shape generator
    constants = flatten(Vector{Float32}, [rand(Float32, 2, 4, 1, 1), rand(Float32, 2, 4, 1, 1)])
    set_constants(te, constants)
    while te.tree.l.l.shape[1] != 4
        reshape_inference(te, cX)
    end

    #println("OK, now we run")
    print(BBB, " ")
    @time eval_diff_tree_array_cpu(trees[5], constants, operators,  cX, reducer_op,       buffer)
end

BBB = 11

if true
    # buffer .= 666
    # BBB = 2000
    # redo_thing()

    for i in 1:5
        tei = make_tensor_expression(trees[i], cX, operators)
        if !reshape_inference(tei, cX)
            continue
        end
        eval_diff_tree_array_gpu(trees[i], constants, operators,  cX, reducer_op,       buffer)
        println()
    end
    # eval_diff_tree_array_gpu(trees[2], constants, operators,  cX, reducer_op,       buffer)
    # eval_diff_tree_array_gpu(trees[3], constants, operators,  cX, reducer_op,       buffer)
    # eval_diff_tree_array_gpu(trees[4], constants, operators,  cX, reducer_op,       buffer)
    # eval_diff_tree_array_gpu(trees[5], constants, operators,  cX, reducer_op,       buffer)

else
    BBB = Int32(11)
    while BBB < 1000_000
        global BBB
        if BBB < 1000
            BBB = Int32(floor(BBB*1.1))
        else
            BBB = Int32(floor(BBB*1.05))
        end
        if false
            print(BBB, " ")
            xa = rand(20*BBB+1)
            xb = rand(20*BBB+1)
            xc = rand(20*BBB+1)
            @time @view(xa[1:(20*BBB)]) .= @view(xb[1:(20*BBB)]) .* @view(xc[1:(20*BBB)])
        else
            redo_thing()
        end
    end
end

# SOMETIMES THIS WHOLE THING DOESN'T WORK BBBECAUSE SHAPE_INFERENCE IS NOT DETERMINISTIC :'(
# NOW IT WORKS BBBECAUSE WE RESHAPE THE CONSTANTS

# options:

#                  the structured expression  inputs   where to store   working
#                                                      the results      memory
# eval_tree_array_cpu(trees[5], constants, operators,  cX,      results,         buffer)

# the reducer_op is applied for every sample to the result and  the last element of cX (the labels) 
# to obtain a scalar, which is summed over the batch size to return a scalar 
# eval_tree_array_cpu(trees[5], constants, operators,  cX,      reducer_op,       buffer)

# the reducer_op is applied for every sample to obtain a scalar, which is summed over the batch size to return a scalar 
# the derivatives with respect to the final scalar are also written into the second sample in the constants ftl
# eval_diff_tree_array_cpu(trees[5], constants, operators,  cX, reducer_op,       buffer)

nothing