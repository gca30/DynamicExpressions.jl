
module EvaluateTensorsModule

using ..FlattenedTensorListModule:
    FlattenedTensorList
using ..NodeModule:
    AbstractNode,
    AbstractExprNode,
    AbstractTensorExprNode
using ..OperatorEnumModule:
    TensorOperatorEnum


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

# function eval_tree_array_cpu(
#     tree::AbstractTensorExprNode{T,N},
#     cX::FlattenedTensorList{T,N,IXT,AT},
#     constants::FlattenedTensorList{T,N,IXT,AT},
#     operators::TensorOperatorEnum
# ) where {T,N,IXT,AT}

#     if tree.degree == 0
#         if tree.constant
#             # value
#             return constants[tree.feature]
#         else
#             # input
#             return cX[tree.feature]
#         end
#     elseif tree.degree == 1
#         top = operators.unaops[tree.op]
#         inner = eval_tree_array_cpu(tree.l, cX, constants, operator)
#         outer = Array{T, N+1}(undef, tree.shape..., size(inner, N+1))
#         top.op!(inner, outer)
#         return outer
#     elseif tree.degree == 2
#         top = operators.binops[tree.op]
#         inner_l = eval_tree_array_cpu(tree.l, cX, constants, operator)
#         inner_r = eval_tree_array_cpu(tree.r, cX, constants, operator)
#         outer = Array{T, N+1}(undef, tree.shape..., max(size(inner_l, N+1), size(inner_r, N+1)))
#         top.op!(inner_l, inner_r, outer)
#         return outer
#     end

# end

function _forward_eval_diff_tree_array_cpu(
    node::AbstractTensorExprNode{T,N},
    cX::FlattenedTensorList{T,N,IXT,AT},
    constants::FlattenedTensorList{T,N,IXT,AT},
    operators::TensorOperatorEnum,
    output::FlattenedTensorList{T,N,IXT,AT},
    batch_offset::Integer, batch_len
) where {T,N,IXT,AT}
    if tree.degree == 0
        return
    elseif tree.degree == 1
        top = operators.unaops[tree.op]
        _forward_eval_diff_tree_array_cpu(node.l, cX, constants, operators, output, batch_offset, batch_len)
        for i in 1:batch_len
            top.op!(
                if node.l.constant 
                    constants[node.l.feature, 1]
                elseif node.l.degree == 0
                    cX[node.l.feature, batch_offset+i]
                else
                    output[node.l.feature, i]
                end, 
                output[node.feature, i]
            )
        end
    elseif tree.degree == 2
        top = operators.unaops[tree.op]
        _forward_eval_diff_tree_array_cpu(node.l, cX, constants, operators, output, batch_offset, batch_len)
        _forward_eval_diff_tree_array_cpu(node.r, cX, constants, operators, output, batch_offset, batch_len)
        for i in 1:batch_len
            top.op!(
                if node.l.constant 
                    constants[node.l.feature, 1]
                elseif node.l.degree == 0
                    cX[node.l.feature, batch_offset+i]
                else
                    output[node.l.feature, i]
                end, 
                if node.r.constant 
                    constants[node.r.feature, 1]
                elseif node.r.degree == 0
                    cX[node.r.feature, batch_offset+i]
                else
                    output[node.r.feature, i]
                end,
                output[node.feature, i]
            )
        end
    end
end

function _backward_eval_diff_tree_array_cpu(
    node::AbstractTensorExprNode{T,N},
    cX::FlattenedTensorList{T,N,IXT,AT},
    constants::FlattenedTensorList{T,N,IXT,AT},
    operators::TensorOperatorEnum,
    output::FlattenedTensorList{T,N,IXT,AT},
    batch_offset::Integer, batch_len
) where {T,N,IXT,AT}
    if tree.degree == 0
        if tree.constant
            for i in 1:batch_len
                @. constants[tree.feature, 2] += output[tree.grad_ix, i] / convert(T, cX.B)
            end
        end
    elseif tree.degree == 1
        top = operators.unaops[tree.op]
        for i in 1:batch_len
            top.grad!(
                output[node.feature, i],
                output[node.grad_ix, i],
                if node.l.degree == 0 && node.l.constant 
                    constants[node.l.feature, 1]
                elseif node.l.degree == 0
                    cX[node.l.feature, i]
                else
                    output[node.l.feature, i]
                end, 
                output[node.l.grad_ix, i]
            )
        end
        _backward_eval_diff_tree_array_cpu(node.l, cX, constants, operators, output, batch_offset, batch_len)
    elseif tree.degree == 2
        top = operators.unaops[tree.op]
        top.grad!(
            output[node.feature, i],
                output[node.grad_ix, i],
            if node.l.degree == 0 && node.l.constant 
                constants[node.l.feature, 1]
            elseif node.l.degree == 0
                cX[node.l.feature, batch_offset+i]
            else
                output[node.l.feature, i]
            end,
            if node.l.has_constants
                output[node.l.grad_ix, i]
            else
                output[1, i]
            end,
            if node.r.degree == 0 && node.r.constant 
                constants[node.r.feature, 1]
            elseif node.r.degree == 0
                cX[node.r.feature, batch_offset+i]
            else
                output[node.r.feature, i]
            end,
            if node.r.has_constants
                output[node.r.grad_ix, i]
            else
                output[1, i]
            end,
            Val((Int8(node.l.has_constants) << 1) | Int8(node.r.has_constants))
        )
        if node.l.has_constants
            _backward_eval_diff_tree_array_cpu(node.l, cX, constants, operators, output, batch_offset, batch_len)
        end
        if node.r.has_constants
            _backward_eval_diff_tree_array_cpu(node.r, cX, constants, operators, output, batch_offset, batch_len)
        end
    end
end

function _eval_diff_tree_array_cpu(
    tree::AbstractTensorExprNode{T,N},
    cX::FlattenedTensorList{T,N,IXT,AT},
    constants::FlattenedTensorList{T,N,IXT,AT},
    operators::TensorOperatorEnum,
    loss_op::Integer,
    output::FlattenedTensorList{T,N,IXT,AT},
    batch_offset, batch_len
) where {T,N,IXT,AT}

    _forward_eval_diff_tree_array_cpu(tree, cX, constants, operators, output, batch_offset, batch_len)
    top = operators.binops[loss_op]
    bc = buffer_count(tree)
    li = loss_gradient_index_in_buffer(tree)
    for i in 1:batch_len
        top.op!(
            if tree.degree == 0 && tree.constant 
                constants[tree.feature, 1]
            elseif tree.degree == 0
                cX[tree.feature, batch_offset + i]
            else
                output[tree.feature, i]
            end, 
            cX[length(cX.positions), batch_offset+i], 
            output[1, i]
        )
        if li <= bc
            top.gradient!(
                output[1, i],
                output[li, i],
                if tree.degree == 0 && tree.constant 
                    constants[tree.feature, 1]
                elseif tree.degree == 0
                    cX[tree.feature, i]
                else
                    output[tree.feature, i]
                end,
                output[tree.grad_ix, i],
                cX[length(cX.positions), batch_offset+i],
                cX[length(cX.positions), batch_offset+i],
                Val(0b11)
            )
        end
    end
    if li <= bc
        _backward_eval_diff_tree_array_cpu(tree, cX, constants, operators, output, batch_offset, batch_len)
    end

end

function eval_diff_tree_array_cpu(
    tree::AbstractTensorExprNode{T,N},
    cX::FlattenedTensorList{T,N,IXT,AT},
    constants::FlattenedTensorList{T,N,IXT,AT},
    operators::TensorOperatorEnum,
    loss_op::Integer,
    buffer::AbstractVector{T}
) where {T,N,IXT,AT}
    
    recalculate_node_values!(tree, constants)
    bc = buffer_count(tree)
    li = loss_gradient_index_in_buffer(tree)
    tempsizes = Vector{NTuple{N,Int32}}(undef, bc)
    tempsizes[1] = ntuple(Returns(1), Val(N))
    if li <= bc
        tempsizes[li] = ntuple(Returns(1), Val(N))
    end
    function recurse_set(node)
        if degree == 2
            recurse_set(node.l)
            recurse_set(node.r)
            tempsizes[node.feature] = node.shape
            if node.constant
                tempsizes[node.grad_ix] = node.shape
            end
        elseif node.degree == 1
            recurse_set(node.l)
            tempsizes[node.feature] = node.shape
            if node.constant
                tempsizes[node.grad_ix] = node.shape
            end
        else
        end
    end
    recurse_set(tree)
    output = treat_as_flattened(buffer, tempsizes, cX.B)

    for i in eachindex(constants.positions)
        @. constants[i, 2] = zero(T)
    end

    for i in 1:div(cX.B + output.B-1, output.B)
        batch_offset = i*output.B
        batch_len = min(cX.B - batch_offset, output.B)
        _eval_diff_tree_array_cpu(tree, cX, constants, operators, loss_op, output, batch_offset, batch_len)
    end

end

function _eval_tree_array_cpu(
    tree::AbstractTensorExprNode{T,N},
    cX::FlattenedTensorList{T,N,IXT,AT},
    constants::FlattenedTensorList{T,N,IXT,AT},
    operators::TensorOperatorEnum,
    loss_op::Integer,
    output::FlattenedTensorList{T,N,IXT,AT},
    batch_offset, batch_len
) where {T,N,IXT,AT}
    _forward_eval_diff_tree_array_cpu(tree, cX, constants, operators, output, batch_offset, batch_len)
    top = operators.binops[loss_op]
    for i in 1:batch_len
        top.op!(
            if tree.degree == 0 && tree.constant 
                constants[tree.feature, 1]
            elseif tree.degree == 0
                cX[tree.feature, batch_offset + i]
            else
                output[tree.feature, i]
            end, 
            cX[length(cX.positions), batch_offset+i], 
            output[1, i]
        )
    end
end

function eval_tree_array_cpu(
    tree::AbstractTensorExprNode{T,N},
    cX::FlattenedTensorList{T,N,IXT,AT},
    constants::FlattenedTensorList{T,N,IXT,AT},
    operators::TensorOperatorEnum,
    loss_op::Integer,
    buffer::AbstractVector{T}
) where {T,N,IXT,AT}
    recalculate_node_values!(tree, constants)
    bc = loss_gradient_index_in_buffer(tree)-1
    tempsizes = Vector{NTuple{N,Int32}}(undef, bc)
    tempsizes[1] = ntuple(Returns(1), Val(N))
    function recurse_set(node)
        if degree == 2
            recurse_set(node.l)
            recurse_set(node.r)
            tempsizes[node.feature] = node.shape
        elseif node.degree == 1
            recurse_set(node.l)
            tempsizes[node.feature] = node.shape
        end
    end
    recurse_set(tree)
    output = treat_as_flattened(buffer, tempsizes, cX.B)

    for i in 1:div(cX.B + output.B-1, output.B)
        batch_offset = i*output.B
        batch_len = min(cX.B - batch_offset, output.B)
        _eval_diff_tree_array_cpu(tree, cX, constants, operators, loss_op, output, batch_offset, batch_len)
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
    recalculate_node_values!(tree, constants)
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