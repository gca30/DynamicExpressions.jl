
module EvaluateTensorsModule

using ..FlattenedTensorListModule:
    FlattenedTensorList,
    treat_as_flattened,
    sample_flat,
    mapself_ti!,
    feature,
    feature_flat,
    value,
    map2_ti!,
    mapk_ti!,
    materialize_ti
using ..NodeModule:
    AbstractNode,
    AbstractExprNode,
    AbstractTensorExprNode
using ..OperatorEnumModule:
    TensorOperatorEnum,
    TensorOperator
using ..TensorNodeUtilsModule:
    buffer_count_with_gradients,
    make_ftl_from_tree
using ..StringsModule:
    string_debug_tree


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

# macro to be only used in the implementations of these functions
#=
macro valsrc(node)
    return quote if $(esc(node)).degree == 0 && $(esc(node)).constant
        $(esc(:constants))[$(esc(node)).feature, 1]
    elseif $(esc(node)).degree == 0
        $(esc(:cX))[$(esc(node)).feature, $(esc(:batch_offset))+$(esc(:i))]
    else
        $(esc(:temp))[$(esc(node)).feature, $(esc(:i))]
    end end
end
=#

cX2 = nothing
function checkme(cX, node)
    global cX2
    if cX2 === nothing
        error("Not set cX2")
    end
    for i in eachindex(cX2.positions)
        if materialize_ti(feature(cX, i)) != materialize_ti(feature(cX2, i))
            error("you changed $(i) when evaluating node $(string_debug_tree(node))")
        end
    end
end

macro valsrc(node)
    return quote if $(esc(node)).degree == 0 && $(esc(node)).constant
        #println("INPUT SOURCE constants")
        $(esc(:feature))($(esc(:constants)), $(esc(node)).feature, 1:1)
    elseif $(esc(node)).degree == 0
        #println("INPUT SOURCE cX")
        $(esc(:feature))($(esc(:cX)), $(esc(node)).feature, ($(esc(:batch_offset))+1):($(esc(:batch_offset))+$(esc(:batch_len))))
    else
        #println("INPUT SOURCE temp")
        $(esc(:feature))($(esc(:temp)), $(esc(node)).feature, 1:$(esc(:batch_len)))
    end end
end

@generated function dispatch_deg1_forward_eval(
    node::AbstractTensorExprNode{T,N},
    constants::FlattenedTensorList{T,N},
    operators::TensorOperatorEnum{NB,BINOPS,NU,UNAOPS},
    cX::FlattenedTensorList{T,N},
    temp::FlattenedTensorList{T,N},
    batch_offset::Integer, batch_len,
) where {T,N,NB,BINOPS,NU,UNAOPS}
    return quote
        # println(string_debug_tree(node.l))
        Base.Cartesian.@nif(
            $NU,
            opix -> opix == node.op,
            opix -> let top = operators.unaops[opix]
                top.op!(
                    feature(temp, node.feature, 1:batch_len), # res
                    @valsrc(node.l) # left
                )
            end
        )
    end
end

@generated function dispatch_deg2_forward_eval(
    node::AbstractTensorExprNode{T,N},
    constants::FlattenedTensorList{T,N},
    operators::TensorOperatorEnum{NB,BINOPS,NU,UNAOPS},
    cX::FlattenedTensorList{T,N},
    temp::FlattenedTensorList{T,N},
    batch_offset::Integer, batch_len,
) where {T,N,NB,BINOPS,NU,UNAOPS}
    return quote
        # println("BINARY NODE:")
        # print(string_debug_tree(node.r))
        # print(string_debug_tree(node.l))
        Base.Cartesian.@nif(
            $NB,
            opix -> opix == node.op,
            opix -> let top = operators.binops[opix]
                top.op!(
                    feature(temp, node.feature, 1:batch_len), # res
                    @valsrc(node.l), # left
                    @valsrc(node.r) # right
                )
            end
        )
    end
end

@generated function dispatch_deg1_backward_eval(
    node::AbstractTensorExprNode{T,N},
    constants::FlattenedTensorList{T,N},
    operators::TensorOperatorEnum{NB,BINOPS,NU,UNAOPS},
    cX::FlattenedTensorList{T,N},
    temp::FlattenedTensorList{T,N},
    batch_offset::Integer, batch_len,
) where {T,N,NB,BINOPS,NU,UNAOPS}
    return quote
        # println(string_debug_tree(node.l))
        Base.Cartesian.@nif(
            $NU,
            opix -> opix == node.op,
            opix -> let top = operators.unaops[opix]
                top.gradient!(
                    feature(temp, node.feature, 1:batch_len), # res
                    feature(temp, node.grad_ix, 1:batch_len), # dres
                    @valsrc(node.l), # left
                    feature(temp, node.l.grad_ix, 1:batch_len) # dleft
                )
            end
        )
    end
end

@generated function dispatch_deg2_backward_eval(
    node::AbstractTensorExprNode{T,N},
    constants::FlattenedTensorList{T,N,IXT},
    operators::TensorOperatorEnum{NB,BINOPS,NU,UNAOPS},
    cX::FlattenedTensorList{T,N},
    temp::FlattenedTensorList{T,N},
    batch_offset::Integer, batch_len,
) where {T,N,IXT,NB,BINOPS,NU,UNAOPS}
    return quote
        # println("BINARY NODE:")
        # print(string_debug_tree(node.r))
        # print(string_debug_tree(node.l))
        Base.Cartesian.@nif(
            $NB,
            opix -> opix == node.op,
            opix -> let top = operators.binops[opix], v = (Int8(node.l.constant) << 1) | Int8(node.r.constant)
                Base.Cartesian.@nif(
                    3,
                    i -> v == i,
                    i -> top.gradient!(
                        feature(temp, node.feature, 1:batch_len), # res
                        feature(temp, node.grad_ix, 1:batch_len), # dres
                        @valsrc(node.l), # left
                        if node.l.constant
                            feature(temp, node.l.grad_ix, 1:batch_len)
                        else
                            feature(temp, 1, 1:batch_len)
                        end, # dleft
                        @valsrc(node.r), # right
                        if node.r.constant
                            feature(temp, node.r.grad_ix, 1:batch_len)
                        else
                            feature(temp, 1, 1:batch_len)
                        end, # dright
                        Val(i) # compute flags
                    )
                )
            end
        )
    end
end

function _forward_eval_diff_tree_array_cpu(
    node::AbstractTensorExprNode{T,N},
    constants::FlattenedTensorList{T,N,IXT,AT},
    operators::TensorOperatorEnum,
    cX::FlattenedTensorList{T,N,IXT,AT},
    temp::FlattenedTensorList{T,N,IXT,AT},
    batch_offset::Integer, batch_len
) where {T,N,IXT,AT}
    #print("FORWARD EVAL OF $(node.index):\n", string_debug_tree(node, operators; indent="    "))
    if node.degree == 0
        return nothing
    elseif node.degree == 1
        _forward_eval_diff_tree_array_cpu(node.l, constants, operators, cX, temp, batch_offset, batch_len)
        dispatch_deg1_forward_eval(node, constants, operators, cX, temp, batch_offset, batch_len)
        #checkme(cX, node)
     #   println("GOT AS FIRST VALUE ", materialize_ti(feature(temp, node.feature, 1:20)))
    elseif node.degree == 2
        _forward_eval_diff_tree_array_cpu(node.l, constants, operators, cX, temp, batch_offset, batch_len)
        _forward_eval_diff_tree_array_cpu(node.r, constants, operators, cX, temp, batch_offset, batch_len)
        # @show node.l
        # @show node.r
        dispatch_deg2_forward_eval(node, constants, operators, cX, temp, batch_offset, batch_len)
        #checkme(cX, node)
      #  println("GOT AS FIRST VALUE ", materialize_ti(feature(temp, node.feature, 1:20)))
    end
    return nothing
end

function _backward_eval_diff_tree_array_cpu(
    node::AbstractTensorExprNode{T,N},
    constants::FlattenedTensorList{T,N,IXT,AT},
    operators::TensorOperatorEnum,
    cX::FlattenedTensorList{T,N,IXT,AT},
    temp::FlattenedTensorList{T,N,IXT,AT},
    batch_offset::Integer, batch_len
) where {T,N,IXT,AT}
   # print("BACKWARDS EVAL OF $(node.index):\n", string_debug_tree(node, operators; indent= "   "))
    if node.degree == 0
        if node.constant
            f1 = feature(constants, node.feature, 2)
            for i in 1:batch_len
                r = feature(temp, node.grad_ix, i)
                map2_ti!(+, f1, f1, r)
            end
        end
    elseif node.degree == 1
        dispatch_deg1_backward_eval(node, constants, operators, cX, temp, batch_offset, batch_len)
        #checkme(cX, node)
    #    println("GOT AS FIRST VALUE ", materialize_ti(feature(temp, node.feature, 1:20)))
        _backward_eval_diff_tree_array_cpu(node.l, constants, operators, cX, temp, batch_offset, batch_len)
    elseif node.degree == 2
        dispatch_deg2_backward_eval(node, constants, operators, cX, temp, batch_offset, batch_len)
        #checkme(cX, node)
     #   println("GOT AS FIRST VALUE ", materialize_ti(feature(temp, node.feature, 1:20)))
        if node.l.constant
            _backward_eval_diff_tree_array_cpu(node.l, constants, operators, cX, temp, batch_offset, batch_len)
        end
        if node.r.constant
            _backward_eval_diff_tree_array_cpu(node.r, constants, operators, cX, temp, batch_offset, batch_len)
        end
    end
end

function _eval_diff_tree_array_cpu(
    tree::AbstractTensorExprNode{T,N},
    constants::FlattenedTensorList{T,N,IXT,AT},
    operators::TensorOperatorEnum,
    cX::FlattenedTensorList{T,N,IXT,AT},
    reducer_op::TensorOperator,
    temp::FlattenedTensorList{T,N,IXT,AT},
    batch_offset, batch_len
) where {T,N,IXT,AT}

    # forward evaluation
    _forward_eval_diff_tree_array_cpu(tree, constants, operators, cX, temp, batch_offset, batch_len)
    
    # reducer operator forward evaluation
    reducer_op.op!(
        feature(temp, 1, 1:batch_len), # res
        @valsrc(tree), # left
        feature(cX, length(cX.positions), (batch_offset+1):(batch_offset+batch_len)) # right
    )

    # reducer operator gradient evaluation
    if tree.constant
        reducer_op.gradient!(
            feature(temp, 1, 1:batch_len), # res
            @valsrc(tree), # left
            feature(temp, tree.grad_ix, 1:batch_len), # dleft
            feature(cX, length(cX.positions), (batch_offset+1):(batch_offset+batch_len)) # right
        )
    end

    result = zero(T)
    for i in 1:batch_len
        result += value(feature_flat(temp, 1, i), 1)
    end

    # backwards gradient evaluation
    if tree.constant
        _backward_eval_diff_tree_array_cpu(tree, constants, operators, cX, temp, batch_offset, batch_len)
    end

    return result
end

function eval_diff_tree_array_cpu(
    tree::AbstractTensorExprNode{T,N},
    constants::FlattenedTensorList{T,N,IXT,AT,APT},
    operators::TensorOperatorEnum,
    cX::FlattenedTensorList{T,N,IXT,AT,APT},
    reducer_op::TensorOperator,
    buffer::AbstractVector{T}
) where {T,N,IXT,AT,APT}
    
    global cX2
    cX2 = FlattenedTensorList(cX.B, cX.L, copy(cX.flattened), cX.positions)
    # println(string_debug_tree(tree, operators))
    output = make_ftl_from_tree(tree, buffer, cX.B, Val(true))
    # display(output)
    # println()

    mapself_ti!(Returns(zero(IXT)), sample_flat(constants, 2))

    result = zero(T)
    for i in 1:div(cX.B + output.B-1, output.B)
        # @show i
        batch_offset = (i-1)*output.B
        batch_len = min(cX.B - batch_offset, output.B)
        result += _eval_diff_tree_array_cpu(tree, constants, operators, cX, reducer_op, output, batch_offset, batch_len)
    end
    return result

end

function _eval_tree_array_cpu(
    tree::AbstractTensorExprNode{T,N},
    constants::FlattenedTensorList{T,N,IXT,AT},
    operators::TensorOperatorEnum,
    cX::FlattenedTensorList{T,N,IXT,AT},
    reducer_op::TensorOperator,
    temp::FlattenedTensorList{T,N,IXT,AT},
    batch_offset, batch_len
) where {T,N,IXT,AT}

    _forward_eval_diff_tree_array_cpu(tree, constants, operators, cX, temp, batch_offset, batch_len)
    
    result = zero(T)
    reducer_op.op!(
        feature(temp, 1, 1:batch_len), # res
        @valsrc(tree), # left
        feature(cX, length(cX.positions), (batch_offset+1):(batch_offset+batch_len)) # right
    )
    for i in 1:batch_len
        result += value(feature_flat(temp, 1, i), 1)
    end
    return result
end

function _eval_tree_array_cpu(
    tree::AbstractTensorExprNode{T,N},
    constants::FlattenedTensorList{T,N,IXT,AT},
    operators::TensorOperatorEnum,
    cX::FlattenedTensorList{T,N,IXT,AT},
    out_results::FlattenedTensorList{T,N,IXT,AT},
    temp::FlattenedTensorList{T,N,IXT,AT},
    batch_offset, batch_len
) where {T,N,IXT,AT}

    _forward_eval_diff_tree_array_cpu(tree, constants, operators, cX, temp, batch_offset, batch_len)
    
    mapk_ti!((_, b) -> b, feature(out_results, 1, (batch_offset+1):(batch_offset+batch_len)), @valsrc(tree))

end

function eval_tree_array_cpu(
    tree::AbstractTensorExprNode{T,N},
    constants::FlattenedTensorList{T,N,IXT,AT},
    operators::TensorOperatorEnum,
    cX::FlattenedTensorList{T,N,IXT,AT},
    reducer_op::TensorOperator,
    buffer::AbstractVector{T}
) where {T,N,IXT,AT}

    output = make_ftl_from_tree(tree, buffer, cX.B, Val(false))

    result = zero(T)
    for i in 1:div(cX.B + output.B-1, output.B)
        batch_offset = i*output.B
        batch_len = min(cX.B - batch_offset, output.B)
        result += _eval_tree_array_cpu(tree, constants, operators, cX, reducer_op, output, batch_offset, batch_len)
    end
    return result

end

function eval_tree_array_cpu(
    tree::AbstractTensorExprNode{T,N},
    constants::FlattenedTensorList{T,N,IXT,AT},
    operators::TensorOperatorEnum,
    cX::FlattenedTensorList{T,N,IXT,AT},
    out_results::FlattenedTensorList{T,N,IXT,AT},
    buffer::AbstractVector{T} 
) where {T,N,IXT,AT}

    output = make_ftl_from_tree(tree, buffer, cX.B, Val(false))
    for i in 1:div(cX.B + output.B-1, output.B)
        batch_offset = i*output.B
        batch_len = min(cX.B - batch_offset, output.B)
        _eval_tree_array_cpu(tree, constants, operators, cX, out_results, output, batch_offset, batch_len)
    end

end




struct GPUInstruction
    
    degree::Int8
    op::Int8
    opv::Int8 # 0 for direct operation, 1 for left derivative, 2 for right derivative, 3 for both left and right derivatives

    l_source::Int8 # 0 for temporary array, 1 for inputs array, 2 for constants array
    r_source::Int8
    
    l_index::Int16
    r_index::Int16
    
    dl_index::Int16
    dr_index::Int16

    result::Int16
    dresult::Int16

    threads::Int32

end

function count_layers(tree::AbstractNode)
    if tree.degree == 1
        return count_layers(tree.l)+1
    elseif tree.degree == 2
        return max(count_layers(tree.l), count_layers(tree.r))+1
    elseif tree.degree == 0
        return 0
    end
end

function count_grad_layers(tree::AbstractNode)
    if !tree.constant
        return 0
    elseif tree.degree == 1
        return count_grad_layers(tree.l)+1
    elseif tree.degree == 2
        return max(count_layers(tree.l), count_layers(tree.r)) + 1
    elseif tree.degree == 0
        return 1
    end
end


function eval_diff_tree_array_gpu(
    tree::AbstractTensorExprNode{T,N},
    cX::FlattenedTensorList{T,N,IXT,AT},
    constants::FlattenedTensorList{T,N,IXT,AT},
    operators::TensorOperatorEnum,
    loss_op::Integer,
    buffer::AbstractVector{T}
) where {T,N,IXT,AT}

    layers = count_layers(tree)
    grad_layers = count_grad_layers(tree)
    instructions = [GPUInstruction[] for _ in 1:(layers+grad_layers)]

    get_source(node) = node.degree == 0 ? (node.constant ? 2 : 1) : 0
    
    function recurse_layers_up(node::AbstractNode)
        layer = if node.degree == 1
            recurse_layers(node.l)+1
        elseif node.degree == 2
            max(recurse_layers(node.l), recurse_layers(node.r))+1
        elseif node.degree == 0
            0
        end
        if node.degree != 0
            instructions[layer].push!(GPUInstruction(
                node.degree, node.op, 0,
                get_source(node.l), node.degree == 2 ? get_source(node.r) : 0, 
                node.l.feature, node.degree == 2 ? node.r.feature : 0, 0, 0,
                node.feature, 0,
                100
            ))
        end
        return layer
    end
    function recurse_layers_down(node::AbstractNode)
        if !node.constant
            return 0
        end
        grad_layer = if node.degree == 1
            count_grad_layers(node.l)+1
        elseif node.degree == 2
            max(count_layers(node.l), count_layers(node.r)) + 1
        elseif node.degree == 0
            1
        end
        if node.degree == 1
            instructions[layers+grad_layers+2-grad_layer].push!(GPUInstruction(
                node.degree, node.op, 1,
                get_source(node.l), 0, 
                node.l.feature, 0, node.l.grad_ix, 0,
                node.feature, node.grad_ix,
                100
            ))
        elseif node.degree == 2
            if node.l.constant && node.r.constant
                instructions[layers+grad_layers+2-grad_layer].push!(GPUInstruction(
                    node.degree, node.op, 3,
                    get_source(node.l), 2, 
                    node.l.feature, node.r.feature, node.l.grad_ix, node.r.grad_ix,
                    node.feature, node.grad_ix,
                    100
                ))
            elseif node.l.constant
                instructions[layers+grad_layers+2-grad_layer].push!(GPUInstruction(
                    node.degree, node.op, 1,
                    get_source(node.l), 2, 
                    node.l.feature, node.r.feature, node.l.grad_ix, node.r.grad_ix,
                    node.feature, node.grad_ix,
                    100
                ))
            elseif node.r.constant
                instructions[layers+grad_layers+2-grad_layer].push!(GPUInstruction(
                    node.degree, node.op, 2,
                    get_source(node.l), 2, 
                    node.l.feature, node.r.feature, node.l.grad_ix, node.r.grad_ix,
                    node.feature, node.grad_ix,
                    100
                ))
            end
        elseif node.degree == 0
        end
        return grad_layer
    end
    recurse_layers_up(tree)
    recurse_layers_down(tree)
    instructions[layers+1].push!(GPUInstruction(
        2, loss_op, 0,
        get_source(tree.l), 1,
        tree.feature, length(cX.positions),
        tree.grad_ix, 0,
        1, 0,
        100
    ))
    instructions[layers+1].push!(GPUInstruction(
        2, loss_op, 1,
        get_source(tree.l), 1,
        tree.feature, length(cX.positions),
        tree.grad_ix, 0,
        1, 0,
        100
    ))
    
end

end