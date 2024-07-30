module StringsModule

using ..UtilsModule: deprecate_varmap
using ..OperatorEnumModule: AbstractOperatorEnum, TensorOperatorEnum, TensorOperator
using ..NodeModule: AbstractScalarExprNode, tree_mapreduce, AbstractTensorExprNode, AbstractExprNode

const OP_NAMES = Base.ImmutableDict(
    "safe_log" => "log",
    "safe_log2" => "log2",
    "safe_log10" => "log10",
    "safe_log1p" => "log1p",
    "safe_acosh" => "acosh",
    "safe_sqrt" => "sqrt",
    "safe_pow" => "^",
)

function dispatch_op_name(::Val{deg}, ::Nothing, idx)::Vector{Char} where {deg}
    if deg == 1
        return vcat(collect("unary_operator["), collect(string(idx)), [']'])
    else
        return vcat(collect("binary_operator["), collect(string(idx)), [']'])
    end
end
function dispatch_op_name_short(::Val{deg}, ::Nothing, idx)::Vector{Char} where {deg}
    if deg == 1
        return collect("u$(idx)")
    else
        return collect("b$(idx)")
    end
end
function dispatch_op_name(::Val{deg}, operators::AbstractOperatorEnum, idx) where {deg}
    if deg == 1
        return get_op_name(operators.unaops[idx])::Vector{Char}
    else
        return get_op_name(operators.binops[idx])::Vector{Char}
    end
end
function dispatch_op_name_short(::Val{deg}, operators::TensorOperatorEnum, idx) where {deg}
    if deg == 1
        return get_op_name(operators.unaops[idx])::Vector{Char}
    else
        return get_op_name(operators.binops[idx])::Vector{Char}
    end
end

function get_op_name(op::TensorOperator)
    return Vector{Char}(String(op.symbol_name))
end

@generated function get_op_name(op::F)::Vector{Char} where {F}
    try
        # Bit faster to just cache the name of the operator:
        
        op_s = if F <: TensorOperator
            string(op.symbol_name)
        elseif F <: Broadcast.BroadcastFunction
            string(F.parameters[1].instance) * '.'
        else
            string(F.instance)
        end
        if length(op_s) == 2 && op_s[1] in ('+', '-', '*', '/', '^') && op_s[2] == '.'
            op_s = '.' * op_s[1]
        end
        out = collect(get(OP_NAMES, op_s, op_s))
        return :($out)
    catch
    end
    return quote
        op_s = typeof(op) <: Broadcast.BroadcastFunction ? string(op.f) * '.' : string(op)
        if length(op_s) == 2 && op_s[1] in ('+', '-', '*', '/', '^') && op_s[2] == '.'
            op_s = '.' * op_s[1]
        end
        out = collect(get(OP_NAMES, op_s, op_s))
        return out
    end
end

@inline function strip_brackets(s::Vector{Char})::Vector{Char}
    if first(s) == '(' && last(s) == ')'
        return s[(begin + 1):(end - 1)]
    else
        return s
    end
end

# Can overload these for custom behavior:
needs_brackets(val::Real) = false
needs_brackets(val::AbstractArray) = false
needs_brackets(val::Complex) = true
needs_brackets(val) = true

function string_constant(val)
    if needs_brackets(val)
        '(' * string(val) * ')'
    else
        string(val)
    end
end

function string_constant_tensor(feature, shape)
    str = "c$(feature)"
    if length(shape) == 0
        return str
    end
    str *= "<"  
    for i in eachindex(shape)
        if i != 1 str *= "x" end
        str *= "$(shape[i])"
    end
    str *= ">"
    return str
end

function string_variable(feature, variable_names)
    if variable_names === nothing ||
        feature > lastindex(variable_names) ||
        feature < firstindex(variable_names)
        return 'x' * string(feature)
    else
        return variable_names[feature]
    end
end

# Vector of chars is faster than strings, so we use that.
function combine_op_with_inputs(op, l, r)::Vector{Char}
    if first(op) in ('+', '-', '*', '/', '^', '.')
        # "(l op r)"
        out = ['(']
        append!(out, l)
        push!(out, ' ')
        append!(out, op)
        push!(out, ' ')
        append!(out, r)
        push!(out, ')')
    else
        # "op(l, r)"
        out = copy(op)
        push!(out, '(')
        append!(out, strip_brackets(l))
        push!(out, ',')
        push!(out, ' ')
        append!(out, strip_brackets(r))
        push!(out, ')')
        return out
    end
end
function combine_op_with_inputs(op, l)
    # "op(l)"
    out = copy(op)
    push!(out, '(')
    append!(out, strip_brackets(l))
    push!(out, ')')
    return out
end

"""
    string_tree(
        tree::AbstractScalarExprNode{T},
        operators::Union{AbstractOperatorEnum,Nothing}=nothing;
        f_variable::F1=string_variable,
        f_constant::F2=string_constant,
        variable_names::Union{Array{String,1},Nothing}=nothing,
        # Deprecated
        varMap=nothing,
    )::String where {T,F1<:Function,F2<:Function}

Convert an equation to a string.

# Arguments
- `tree`: the tree to convert to a string
- `operators`: the operators used to define the tree

# Keyword Arguments
- `f_variable`: (optional) function to convert a variable to a string, with arguments `(feature::UInt8, variable_names)`.
- `f_constant`: (optional) function to convert a constant to a string, with arguments `(val,)`
- `variable_names::Union{Array{String, 1}, Nothing}=nothing`: (optional) what variables to print for each feature.
"""
function string_tree(
    tree::AbstractScalarExprNode{T},
    operators::Union{AbstractOperatorEnum,Nothing}=nothing;
    f_variable::F1=string_variable,
    f_constant::F2=string_constant,
    variable_names::Union{AbstractVector{<:AbstractString},Nothing}=nothing,
    # Deprecated
    varMap=nothing,
)::String where {T,F1<:Function,F2<:Function}
    variable_names = deprecate_varmap(variable_names, varMap, :string_tree)
    raw_output = tree_mapreduce(
        let f_constant = f_constant,
            f_variable = f_variable,
            variable_names = variable_names

            (leaf,) -> if leaf.constant
                collect(f_constant(leaf.val))::Vector{Char}
            else
                collect(f_variable(leaf.feature, variable_names))::Vector{Char}
            end
        end,
        let operators = operators
            (branch,) -> if branch.degree == 1
                dispatch_op_name(Val(1), operators, branch.op)::Vector{Char}
            else
                dispatch_op_name(Val(2), operators, branch.op)::Vector{Char}
            end
        end,
        combine_op_with_inputs,
        tree,
        Vector{Char};
        f_on_shared=(c, is_shared) -> if is_shared
            out = ['{']
            append!(out, c)
            push!(out, '}')
            out
        else
            c
        end,
    )
    return String(strip_brackets(raw_output))
end

function string_tree(
    tree::AbstractTensorExprNode{T},
    operators::Union{TensorOperatorEnum,Nothing}=nothing;
    f_variable::F1=string_variable,
    f_constant::F2=string_constant_tensor,
    variable_names::Union{AbstractVector{<:AbstractString},Nothing}=nothing
)::String where {T,F1<:Function,F2<:Function}
    raw_output = tree_mapreduce(
        let f_constant = f_constant,
            f_variable = f_variable,
            variable_names = variable_names

            (leaf,) -> if leaf.constant
                collect(f_constant(leaf.feature, leaf.shape))::Vector{Char}
            else
                collect(f_variable(leaf.feature, variable_names))::Vector{Char}
            end
        end,
        let operators = operators
            (branch,) -> if branch.degree == 1
                dispatch_op_name_short(Val(1), operators, branch.op)::Vector{Char}
            else
                dispatch_op_name_short(Val(2), operators, branch.op)::Vector{Char}
            end
        end,
        combine_op_with_inputs,
        tree,
        Vector{Char};
        f_on_shared=(c, is_shared) -> if is_shared
            out = ['{']
            append!(out, c)
            push!(out, '}')
            out
        else
            c
        end,
    )
    return String(strip_brackets(raw_output))
end

function string_debug_tree_header(
    tree::AbstractTensorExprNode, 
    operators::Union{TensorOperatorEnum, Nothing}, 
    variable_names::Union{AbstractVector{<:AbstractString},Nothing}
)
    s = ""
    if tree.degree == 0
        if tree.constant
            s = String(string_constant_tensor(tree.feature, ()))
        else
            s = String(string_variable(tree.feature, variable_names))
        end
    elseif tree.degree == 1
        s = "unaop " * String(dispatch_op_name_short(Val(1), operators, tree.op))
    elseif tree.degree == 2
        s = "binop " * String(dispatch_op_name_short(Val(2), operators, tree.op))
    end
    s *= " <"
    for i in eachindex(tree.shape)
        if i != 1 s *= "x" end
        s *= "$(tree.shape[i])"
    end
    s *= "> "
    s *= "i$(tree.index) f$(tree.feature) g$(tree.grad_ix)"
    s *= tree.constant ? " cs" : " nc"
end

function string_debug_tree_header(tree::AbstractScalarExprNode, operators::Union{AbstractOperatorEnum, Nothing}, variable_names::Union{AbstractVector{<:AbstractString},Nothing})
    return "NOT YET IMPLEMENTED"
end

function string_debug_tree(
    tree::AbstractExprNode{T},
    operators::Union{AbstractOperatorEnum, Nothing}=nothing,
    f_header = string_debug_tree_header,
    variable_names::Union{AbstractVector{<:AbstractString},Nothing}=nothing;
    indent=""
) where {T}
    function recurse(node, indent1, indent2)
        val = indent1 * f_header(node, operators, variable_names) * "\n"
        if node.degree == 0
        elseif node.degree == 1
            val *= recurse(node.l, indent2 * "└─", indent2 * "  ")
        elseif node.degree == 2
            val *= recurse(node.l, indent2 * "├─", indent2 * "│ ")
            val *= recurse(node.r, indent2 * "└─", indent2 * "  ")
        end
        return val
    end
    return recurse(tree, indent, indent)
end


# Print an equation
for io in ((), (:(io::IO),))
    @eval function print_tree(
        $(io...),
        tree::AbstractScalarExprNode,
        operators::Union{AbstractOperatorEnum,Nothing}=nothing;
        f_variable::F1=string_variable,
        f_constant::F2=string_constant,
        variable_names::Union{AbstractVector{<:AbstractString},Nothing}=nothing,
        # Deprecated
        varMap=nothing,
    ) where {F1<:Function,F2<:Function}
        variable_names = deprecate_varmap(variable_names, varMap, :print_tree)
        return println(
            $(io...), string_tree(tree, operators; f_variable, f_constant, variable_names)
        )
    end
end

end
