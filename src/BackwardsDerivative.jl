
using ..NodeModule: Node

# It's important to allocate the values from the begining as they will be reused a lot
mutable struct DifferentiableNodeValue{T}
    node::Node{T}
    value::T
    has_consts_l::Bool # whether a tree has any constant children on its left side
    has_consts_r::Bool # whether a tree has any constant children on its right side
    derivative_l::T
    derivative_r::T
end

@inline multiply_jacobian(next_der, prev_der, J) = begin
    next_der .= reshape(transpose(J) * reshape(prev_der, length(prev_der)), size(next_der))
end

# An operator needs a lot of 
struct OperatorInfo{Fdirect, Finplace, Fconstraint, Fgradient, Fcomplexity}
    degree::UInt8 # degree of operator
    T::DataType # type it works on
    op::Fdirect # actual operator (T[, T]) -> T
    op!::Finplace # in place operator (T, T[, T]) -> nothing
    gen_constraints::Fconstraint # generate constraints for shape 
    gradient!::Fgradient # in place gradient function (∂out::T, l::T, ∂l::T[, r::T, ∂r::T]) -> nothing
    get_complexity::Fcomplexity # complexity of the operator as a funcition of their shapes

    OperatorInfo(
        degree, T; op,
        op! = if degree == 1
            (out, x) -> @. out = op(x)
        else
            (out, x, y) -> @. out = op(x, y)
        end,
        gen_constraints = if degree == 1
            append_constraints_unary_all_equal
        else
            append_constraints_binary_all_equal
        end,
        gradient! = if degree == 1
            (∂out::T, l::T, ∂l::T) -> multiply_jacobian(∂l, ∂out, jacobian(op, l)[1])
        else
            (∂out::T, l::T, ∂l::T, r::T, ∂r::T) -> begin
                Js = jacobian(op, l, r)
                multiply_jacobian(∂l, ∂out, Js[1])
                multiply_jacobian(∂r, ∂out, Js[2])
            end
        end,
        get_complexity = if degree == 1
            (shape) -> prod(shape)
        else
            (shape_a, shape_b) -> prod(map(maximum, zip(shape_a, shape_b)))
        end
    ) = begin 
        @assert degree == 1 || degree == 2
        @assert T <: AbstractArray
        new(degree, T, op, op!, gen_constraints, gradient!, get_complexity)
    end
end

mutable struct OptimizeFrame{T,BT,LFT,OPT}
    tree::T
    operators::OPT
    value::BT
    root_derivative::T
    loss_func::LFT
    constants::Vector{BT}
    refs::Vector{Ref{Node{T}}}
    index::Dict{Node{T}, DifferentiableNodeValue{T}}
    computed::UInt8
        # 0b001 -> updated the constants from the constants vector
        # 0b010 -> computed value
        # 0b100 -> computed derivative
end

function make_frame(tree::Node{AT}, cX::AbstractVector{ATP1}, shapes::Dict{Node{AT}, NTuple{N, Int64}}, operators::OperatorEnum, allocator::F, loss_function::LFT) where {BT,N,F,AT<:AbstractArray{BT,N},NP1,ATP1<:AbstractArray{BT,NP1},LFT} 
#= ::OptimizeFrame{AT,BT}, with F function from NTuple{N, Int64} to ATP1 or nothing to empty =#
    index = Dict{Node{T}, DifferentiableNodeValue{T}}()
    zeros = ntuple(Returns(0), Val(NP1))
    datapoints = size(cX[1], 1)
    add_node(node::Node{AT}) = begin
        if node.degree == 0
            if node.constant
                index[node] = DifferentiableNodeValue(
                    node, allocator(datapoints, size(node.val)...), true, false, allocator(1, shapes[node]...), allocator(zeros)
                )
            else
                index[node] = DifferentiableNodeValue(
                    node, cX[node.feature], false, false, allocator(zeros), allocator(zeros)
                )
            end
            return node.constant
        elseif node.degree == 1
            has_consts = add_node(node.l) 
            index[node] = DifferentiableNodeValue(
                node, allocator(datapoints, shapes[node]...), has_consts, false, has_consts ? allocator(datapoints, shapes[node.l]...) : allocator(zeros), allocator(zeros)
            )
            return has_consts
        else
            has_consts_l = add_node(node.l)
            has_consts_r = add_node(node.r) 
            index[node] = DifferentiableNodeValue(
                node, allocator(datapoints, shapes[node]...), has_consts_l, has_consts_r, 
                has_consts_l ? allocator(datapoints, shapes[node.l]...) : allocator(), 
                has_consts_r ? allocator(datapoints, shapes[node.r]...) : allocator()
            )
            return has_consts_l || has_consts_r
        end
    end
    constants, refs = get_constants(tree)
    root_derivative = allocator(datapoints, shapes[tree]...)
    OptimizeFrame(tree, operators, zero(BT), root_derivative, loss_function, constants, refs, index, 0)
end

function eval_frame_node(frame::OptimizeFrame{T,BT,LFT}, node::Node{T}) where {T, BT, LFT}
    
    if node.degree == 0
        if node.constant
            # hopefully there won't be an extra gpu allocation here
            index[node].value .= repeat(reshape(node.val, 1, size(node.val)...))
        end
    elseif node.degree == 1
        op = frame.operators.unaops[node.op]
        eval_frame_node(frame, node.l)
        # currenlty when you use this it still allocs memory
        index[node].value .= op(index[node.l].value)
    elseif node.degree == 2
        op = frame.operators.binops[node.op]
        eval_frame_node(frame, node.l)
        eval_frame_node(frame, node.r)
        index[node].value .= op(node.l, node.r)
    end

end

function eval_frame(frame::OptimizeFrame{T,BT,LFT}) where {T,BT,LFT}
    if frame.computed & 0b010 != 0
        return frame.value
    end
    if frame.computed & 0b001 == 0
        set_constants!(tree, constants, refs)
        frame.computed |= 0b001
    end
    eval_frame_node(frame, frame.tree)
    final_value = frame.index[free.tree].value
    frame.value = sum(i -> frame.loss_func(selectdim(final_value, 1, i)), axes(final_value, 1))

    frame.computed |= 0b010
    return frame.value
end



# TODO: it would be a lot, lot better to use these kinds of functions
# where we don't use a massive jacobian and we don't allocate anything
# downside: we have to define derivatives for everything, insted of it being done automatically

actually_good_derivative(::typeof(.+), ∂out, r, ∂r, l, ∂l) = begin
    sum!(∂r, ∂out)
    sum!(∂l, ∂out)
end

actually_good_derivative(::typeof(.-), ∂out, r, ∂r, l, ∂l) = begin
    sum!(∂l, ∂out)
    sum!(-, ∂r, ∂out)
end

actually_good_derivative(::typeof(.*), ∂out, r, ∂r, l, ∂l) = begin
    sum!(∂l, ∂out ./ r)
    map!(x -> is_valid(x) ? x : zero(x), ∂l, ∂l)
    sum!(∂r, ∂out ./ l)
    map!(x -> is_valid(x) ? x : zero(x), ∂r, ∂r)
end

function eval_diff_frame_node(frame::OptimizeFrame{T}, node::Node{T}, derivative::AT) where {T,AT}
    if node.degree == 0
        sum!(frame.index[node].derivative_l, derivative)
    elseif node.degree == 1
        op = frame.operators.unaops[node.op]
        inner_val = frame.index[node.l].value
        multiply_jacobian(frame.index[node].derivative_l, derivative, jacobian(op, inner_val)[1])
        eval_diff_frame_node(frame, node.l, frame.index[node].derivative_l)
    elseif node.degree == 2
        op = frame.operators.binops[node.op]
        op = frame.operators.unaops[node.op]
        val_l = frame.index[node.l].value
        val_r = frame.index[node.r].value
        # Jacobian
        (d_val_l, d_val_r) = jacobian(op, val_l, val_r)
        multiply_jacobian(frame.index[node].derivative_l, derivative, d_val_l)
        multiply_jacobian(frame.index[node].derivative_r, derivative, d_val_r)
        if frame.index[node.l].has_consts_l
            eval_diff_frame_node(frame, node.l, frame.index[node].derivative_l)
        end
        if frame.index[node.r].has_consts_r
            eval_diff_frame_node(frame, node.r, frame.index[node].derivative_r)
        end
    end
end

function set_derivatives(frame::OptimizeFrame{T}, node::Node{T}, idx, gradient_vector) where {T}
    if node.degree == 0
        if node.constant
            derivative = frame.index[node].derivative_l
            copyto!(@view(gradient_vector[idx:end]), reshape(sum(derivative; dims=1)./size(derivative,1), length(derivative)/size(derivative, 1)))
            idx += length(derivative)/size(derivative, 1)
        end
    elseif node.degree == 1
        idx = set_derivatives(frame, node.l, idx, gradient_vector)
    elseif node.degree == 2
        idx = set_derivatives(frame, node.l, idx, gradient_vector)
        idx = set_derivatives(frame, node.r, idx, gradient_vector)
    end
    return idx
end

function eval_diff_frame(frame::OptimizeFrame{T,BT,LFT}, gradient_vector) where {T,BT,LFT}
    if frame.computed & 0b100 != 0
        return gradient_vector
    end
    eval_frame(frame)
    frame.root_gradient = gradient(
        final_value -> sum(i -> frame.loss_func(selectdim(final_value, 1, i)), axes(final_value, 1)),
        frame.index[tree].value
    )[1]
    eval_diff_frame_node(frame, frame.tree, frame.root_derivative)
    set_derivatives(frame, frame.tree, 0, gradient_vector)

    frame.computed |= 0b100
    return frame.derivatives
end

function shape_upgrade(shapes::Dict{Node{AT}, NTuple{N, Int64}}) where {BT,N,AT<:AbstractArray{BT,N}}
    for node in keys(shapes)
        (node.degree != 0 || !node.constant) && continue
        desired_shape = shapes[node]
        current_shape = size(node.val)
        if desired_shape != current_shape
            node.val = AT <: CUDA.CuArray ? CUDA.rand(desired_shape) : rand(desired_shape)
        end
    end
end

function optimize_constants!(tree::Node{AT}, operators::OperatorEnum, cX::AbstractVector{ATP1}, y::AbstractArray{ATP1}, loss_function) where {BT,N,AT<:AbstractArray{BT,N},NP1,ATP1<:AbstractArray{BT,N}}
    allocator = (sizes) -> error("Unkown array type")
    datasets = size(y, 1)
    @show AT
    return
    if AT <: Array{BT,N}
        allocator = (sizes) -> Array{BT,length(sizes)}(undef, sizes...)
    elseif AT <: CUDA.CuArray{BT,N}
        allocator = (sizes) -> CuArray{BT,length(sizes)}(undef, sizes...)
    end
    shapes = ntuple(i -> i > length(cX) ? size(y) : size(cX[i]), length(cX)+1)
    shapes_d, ok = shape_inference(tree, operators, shapes; throw_errors = false)
    !ok && error("Wrong shape!!!")
    shape_upgrade(shapes_d)
    frame = make_frame(tree, cX, shapes_d, operators, allocator, loss_function)
    
    Optim.optimize((new_constants) -> begin
        if new_constants != frame.constants
            frame.constants .= new_constants
            frame.computed = 0
        end
        return eval_frame(frame)
    end,
    (gradient_vector, new_constants) -> begin
        if new_constants != frame.constants
            frame.constants .= new_constants
            frame.computed = 0
        end
        return eval_diff_frame(frame, gradient_vector)
    end,
    frame.constants
    ; inplace=true)
end