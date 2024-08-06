
module TensorExpressionModule

using ..FlattenedTensorListModule: FlattenedTensorList
using ..NodeModule: AbstractTensorExprNode
using ..OperatorEnumModule: TensorOperatorEnum

# Many functions don't work on the AbstractExpression
abstract type AbstractTensorExpression{T,N} end

mutable struct TensorExpression{T,N,FTL<:FlattenedTensorList{T,N},NodeT<:AbstractTensorExprNode{T,N},TOET<:TensorOperatorEnum,MetadataT<:NamedTuple} <: AbstractTensorExpression{T,N}
    tree::NodeT
    constants::FTL
    operators::TOET
    metadata::MetadataT
end

get_tree(te::TensorExpression) = te.tree
get_constants(te::TensorExpression) = te.constants
set_constants(te::TensorExpression, constants::FlattenedTensorList) = te.constants = constants
get_operators(te::TensorExpression) = te.operators
get_metadata(te::TensorExpression) = te.metadata

function make_tensor_expression(tree::AbstractTensorExprNode{T,N}, constants::FlattenedTensorList{T,N}, operators::TensorOperatorEnum) where {T,N}
    return TensorExpression(tree, constants, operators, (;))
end

function Base.copy(te::TensorExpression; break_sharing=Val(false))
    return TensorExpression(copy(te.tree; break_sharing), copy(te.constants), operators, deepcopy(metadata))
end

# get_operators(te::TensorExpression) = te.operators
# get_variable_names(te::TensorExpression, variable_names::Union{Nothing,AbstractVector{<:AbstractString}}=nothing) = variable_names === nothing ? ex.metadata.variable_names : variable_names
# #Base.copy(te::TensorExpression; break_sharing=Val(false)) = TensorExpression(copy(tree; break_sharing), copy(constants; break_sharing), operators, copy(metadata))
# get_scalar_constants(te::TensorExpression) = 0 # TODO
# set_scalar_constants(te::TensorExpression, constants, refs) = 0 # TODO
# extract_gradient(gradient, ex::AbstractExpression) = 0 # TODO
# get_tree(te::TensorExpression) = te.tree
# get_contents(te::TensorExpression) = (te.tree, te.constants, te.operators)
# get_metadata(te::TensorExpression) = te.metadata

end