module AsArrayModule

using ..EquationModule: AbstractExpressionNode, tree_mapreduce, count_nodes

function as_array(
    ::Type{I},
    tree::N,
    additional_trees::Vararg{N,M};
    buffer::Union{AbstractArray,Nothing}=nothing,
) where {T,N<:AbstractExpressionNode{T},I,M}
    trees = (tree, additional_trees...)
    each_num_nodes = (t -> count_nodes(t; break_sharing=Val(true))).(trees)
    num_nodes = sum(each_num_nodes)

    roots = cumsum(tuple(one(I), each_num_nodes[1:(end - 1)]...))

    val = Array{T}(undef, num_nodes)

    ## Views of the same matrix:
    buffer = buffer === nothing ? Array{I}(undef, 8, num_nodes) : buffer
    degree = @view buffer[1, :]
    feature = @view buffer[2, :]
    op = @view buffer[3, :]
    execution_order = @view buffer[4, :]
    idx_self = @view buffer[5, :]
    idx_l = @view buffer[6, :]
    idx_r = @view buffer[7, :]
    constant = @view buffer[8, :]

    cursor = Ref(zero(I))
    for (root, tree) in zip(roots, trees)
        @assert root == cursor[] + 1
        tree_mapreduce(
            leaf -> begin
                self = (cursor[] += one(I))
                idx_self[self] = self
                degree[self] = 0
                execution_order[self] = one(I)
                constant[self] = leaf.constant
                if leaf.constant
                    val[self] = leaf.val::T
                else
                    feature[self] = leaf.feature
                end

                (id=self, order=one(I))
            end,
            branch -> begin
                self = (cursor[] += one(I))
                idx_self[self] = self
                op[self] = branch.op
                degree[self] = branch.degree

                (id=self, order=one(I))  # this order is unused
            end,
            ((parent, children::Vararg{Any,C}) where {C}) -> begin
                idx_l[parent.id] = children[1].id
                if C == 2
                    idx_r[parent.id] = children[2].id
                end
                parent_execution_order = if C == 1
                    children[1].order + one(I)
                else
                    max(children[1].order, children[2].order) + one(I)
                end
                execution_order[parent.id] = parent_execution_order

                (id=parent.id, order=parent_execution_order)
            end,
            tree;
            break_sharing=Val(true),
        )
    end

    return (;
        degree,
        constant,
        val,
        feature,
        op,
        execution_order,
        idx_self,
        idx_l,
        idx_r,
        roots,
        buffer,
        num_nodes,
    )
end

end
