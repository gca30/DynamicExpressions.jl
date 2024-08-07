module TensorNodeUtilsModule

import Compat: Returns
import ..NodeModule:
    AbstractNode,
    AbstractExprNode,
    AbstractTensorExprNode,
    Node,
    preserve_sharing,
    constructorof,
    copy_node,
    count_nodes,
    tree_mapreduce,
    any,
    filter_map
import ..FlattenedTensorListModule:
    FlattenedTensorList,
    permute_features,
    treat_as_flattened,
    feature_flat,
    copyto_ti!,
    mapk_ti!,
    feature,
    selectdim_ti
using ..NodeUtilsModule:
    count_constant_nodes,
    is_node_constant

function recalculate_constant_indices(tree::AbstractTensorExprNode{T,N}, constants::FlattenedTensorList{T,N}) where {T,N}
    function recurse(node, cindex, v)
        if node.degree == 2
            cindex = recurse(node.l, cindex, v)
            cindex = recurse(node.r, cindex, v)
        elseif node.degree == 1
            cindex = recurse(node.l, cindex, v)
        elseif node.degree == 0 && node.constant
            v[cindex] = node.feature
            node.feature = cindex
            return cindex + 1
        end
        return cindex
    end
    count_consts = count_constant_nodes(tree)
    v = Vector{Int32}(undef, count_consts)
    recurse(tree, 1, v)
    for i in eachindex(v)
        if i != v[i]
            return permute_features(constants, v)
        end
    end
    return constants
end

# renumbers the nodes to be from 1 to the number of temporary nodes (meaning inputs and constants are not numbered)
# this removes the information of the original constant indices
function recalculate_node_indices!(tree::AbstractTensorExprNode)
    function recurse(node, nfeature, nindex) 
        if node.degree == 2
            nfeature, nindex = recurse(node.l, nfeature, nindex)
            nfeature, nindex = recurse(node.r, nfeature, nindex)
            node.feature = nfeature
            node.index = nindex
            return nfeature+1, nindex+1
        elseif node.degree == 1
            nfeature, nindex = recurse(node.l, nfeature, nindex)
            node.feature = nfeature
            node.index = nindex
            return nfeature+1, nindex+1
        elseif node.degree == 0
            node.index = nindex
            return nfeature, nindex+1
        end
    end
    recurse(tree, 2, 1)
end

# a the nodes to be from 1 to the number of temporary nodes (meaning inputs and constants are not numbered)
# this removes the information of the original constant indices
@inline buffer_count_without_gradients(tree) = tree_mapreduce((n -> n.degree == 0 ? 1 : n.feature), max, tree)
@inline buffer_count_with_gradients(tree) = tree_mapreduce((n -> max(n.constant ? n.grad_ix : 1, n.degree == 0 ? 1 : n.feature)), max, tree)
@inline number_of_indices(tree) = tree_mapreduce((n -> n.index), max, tree)

function recalculate_constant!(tree::AbstractTensorExprNode)
    if tree.degree == 2
        recalculate_constant!(tree.l)
        recalculate_constant!(tree.r)
        tree.constant = tree.l.constant || tree.r.constant
    elseif tree.degree == 1
        recalculate_constant!(tree.l)
        tree.constant = tree.l.constant
    end
end

function recalculate_gradient_indices!(tree::AbstractTensorExprNode)
    b = buffer_count_without_gradients(tree)
    # maxf - maximum feature so far, 1 if no features
    function recurse(node, ix)
        if !node.constant
            node.grad_ix = 0
            return ix 
        end
        if node.degree == 1
            ix = recurse(node.l, ix)
        elseif node.degree == 2
            ix = recurse(node.l, ix)
            ix = recurse(node.r, ix)
        end
        node.grad_ix = ix
        return ix+1
    end
    recurse(tree, b+1)
end

function recalculate_node_values!(tree::AbstractTensorExprNode{T,N}, constants::FlattenedTensorList{T,N}) where {T,N}
    new_constants = recalculate_constant_indices(tree, constants)
    recalculate_constant!(tree)
    recalculate_node_indices!(tree)
    recalculate_gradient_indices!(tree)
    return new_constants
end

function make_ftl_from_tree(tree::AbstractTensorExprNode{T,N}, buffer::AbstractVector{T}, maxB, ::Val{with_gradients}) where {T,N,with_gradients}
    bc = with_gradients ? buffer_count_with_gradients(tree) : buffer_count_without_gradients(tree)
    tempsizes = Vector{NTuple{N,Int32}}(undef, bc)
    tempsizes[1] = ntuple(Returns(1), Val(N))
    function recurse_set(node)
        if node.degree == 2
            recurse_set(node.l)
            recurse_set(node.r)
            tempsizes[node.feature] = node.shape
        elseif node.degree == 1
            recurse_set(node.l)
            tempsizes[node.feature] = node.shape
        end
        if with_gradients && node.constant
            tempsizes[node.grad_ix] = node.shape
        end
    end
    recurse_set(tree)
    return treat_as_flattened(buffer, tempsizes, maxB)
end


function reshape_constants(tree::AbstractTensorExprNode{T,N}, constants::FlattenedTensorList{T,N,IXT,AT}) where {T,N,IXT,AT}
    all_ok = tree_mapreduce(
        node -> is_node_constant(node) ? constants.positions[node.feature].shape == node.shape : true,
        &, tree, Bool
    )
    if all_ok return constants end
    v = Vector{NTuple{N,IXT}}(undef, count_constant_nodes(tree))
    tree_mapreduce(
        node -> begin
            if is_node_constant(node)
                v[node.feature] = node.shape
            end
            return nothing
        end,
        Returns(nothing),
        tree, Nothing
    )
    B = constants.B
    buf = AT(undef, B*sum(prod, v))
    consts2 = treat_as_flattened(buf, v, B)
    for ci in eachindex(v)
        new_shape = consts2.positions[ci].shape
        new_len = consts2.positions[ci].len
        old_shape = constants.positions[ci].shape 
        old_len = constants.positions[ci].len
        if new_shape == old_shape
            mapk_ti!((_,b)->b, feature(consts2, ci), feature(constants, ci))
        elseif new_len == old_len
            mapk_ti!((_,b)->b, feature_flat(consts2, ci), feature_flat(constants, ci))
        elseif new_len < old_len
            mapk_ti!((_,b)->b, feature_flat(consts2, ci), selectdim_ti(feature_flat(constants, ci), 1, 1:new_len))
        else
            off = 0
            while off <= new_len
                if off+old_len > new_len
                    mapk_ti!((_,b)->b, 
                        selectdim_ti(feature_flat(consts2, ci), 1, (off+1):(new_len)), 
                        selectdim_ti(feature_flat(constants, ci), 1, 1:(new_len-off))
                    )
                else
                    mapk_ti!((_,b)->b, 
                        selectdim_ti(feature_flat(consts2, ci), 1, (off+1):(off+old_len)), 
                        feature_flat(constants, ci)
                    )
                end
                off += old_len
            end
        end
        
        # ideally it would work like this:
        # first rule: if they have the same amount of elements, just keep them (equivalent to julia `reshape` function)
        # second rule: if they are broadcastable, broadcast them
        # if all(((n, o),) -> n == o || n == 1 || o == 1, zip(new_shape, old_shape))
        #     # should reduce?
        #     if any(((n, o),) -> n == 1 && o != 1, zip(new_shape, old_shape))
        #         consts2[ci] .= sum(constants[ci]; dims=
        #             filter(i -> new_shape[i] == 1 && old_shape[i] != 1, ntuple(i -> i, Val(N)))
        #         )
        #     else
        #         consts2[ci] .= constants[ci]
        #     end
        # end
        # third rule: get what dimensions are broadcastable, but clamp the rest
        
    end
    return consts2
end

end
