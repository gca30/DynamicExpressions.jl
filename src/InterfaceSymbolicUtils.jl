using FromFile
using SymbolicUtils
@from "Core.jl" import CONST_TYPE, Node, Options
@from "Utils.jl" import isgood, isbad, @return_on_false

const SYMBOLIC_UTILS_TYPES = Union{<:Number,SymbolicUtils.Sym{<:Number},SymbolicUtils.Term{<:Number}}

"""
    node_to_symbolic(tree::Node, options::Options;
                varMap::Union{Array{String, 1}, Nothing}=nothing,
                evaluate_functions::Bool=false,
                index_functions::Bool=false)

The interface to SymbolicUtils.jl. Passing a tree to this function
will generate a symbolic equation in SymbolicUtils.jl format.

## Arguments

- `tree::Node`: The equation to convert.
- `options::Options`: Options, which contains the operators used in the equation.
- `varMap::Union{Array{String, 1}, Nothing}=nothing`: What variable names to use for
    each feature. Default is [x1, x2, x3, ...].
- `evaluate_functions::Bool=false`: Whether to evaluate the operators, or
    leave them as symbolic.
- `index_functions::Bool=false`: Whether to generate special names for the
    operators, which then allows one to convert back to a `Node` format
    using `symbolic_to_node`.
"""



const SUPPORTED_OPS = (cos, sin, exp, cot, tan, csc, sec, +, -, *, /)

isgood(x::SymbolicUtils.Symbolic) = SymbolicUtils.istree(x) ? all(isgood.([SymbolicUtils.operation(x);SymbolicUtils.arguments(x)])) : true
subs_bad(x) = isgood(x) ? x : Inf

function parse_tree_to_eqs(tree::Node, opts, index_functions = false, evaluate_functions = false)
    if tree.degree == 0
        # Return constant if needed
        tree.constant && return subs_bad(tree.val)
        return SymbolicUtils.Sym{Real}(Symbol("x$(tree.feature)"))
    end
    # Collect the next children
    children = tree.degree >= 2 ? (tree.l, tree.r) : (tree.l,)
    # Get the operation
    op = tree.degree > 1 ? opts.binops[tree.op] : opts.unaops[tree.op]
    # Create an N tuple of Numbers for each argument
    dtypes = map(x->Number, 1:tree.degree)
    #
    if index_functions
        symname = tree.degree > 1 ? "_binops$(op)" : "_unaops$(op)"
        op = SymbolicUtils.Sym{(SymbolicUtils.FnType){Tuple{dtypes...}, Real}}(Symbol(symname))
    else
        op = (op ∈ SUPPORTED_OPS) || evaluate_functions ? op : SymbolicUtils.Sym{(SymbolicUtils.FnType){Tuple{dtypes...}, Real}}(Symbol(op))
    end
    return subs_bad(op(map(x->parse_tree_to_eqs(x, opts, index_functions, evaluate_functions), children)...))
end


function node_to_symbolic(tree::Node, options::Options;
                     varMap::Union{Array{String, 1}, Nothing}=nothing,
                     evaluate_functions::Bool=false,
                     index_functions::Bool=false
                     )
    expr = subs_bad(parse_tree_to_eqs(tree, options, index_functions, evaluate_functions))
    # Check for NaN and Inf
    @assert isgood(expr) "The recovered equation contains NaN or Inf."
    # Return if no varMap is given
    isnothing(varMap) && return expr
    # Create a substitution tuple
    subs = Dict(
        [SymbolicUtils.Sym{Real}(Symbol("x$(i)")) => SymbolicUtils.Sym{Real}(Symbol(varMap[i])) for i in 1:length(varMap)]...
    )
    return substitute(expr, subs)
end



# Just constant
function symbolic_to_node(eqn::T, options::Options;
                     varMap::Union{Array{String, 1}, Nothing}=nothing)::Node where {T<:Number}
    return Node(convert(CONST_TYPE, eqn))
end

# Just variable
function symbolic_to_node(eqn::T, options::Options;
                     varMap::Union{Array{String, 1}, Nothing}=nothing)::Node where {T<:SymbolicUtils.Sym{<:Number}}
    return Node(varMap_to_index(eqn.name, varMap))
end

function _multiarg_split(op_idx::Int, eqn::Array{Any, 1},
                        options::Options, varMap::Union{Array{String, 1}, Nothing}
                       )::Node
    if length(eqn) == 2
        return Node(op_idx,
                    symbolic_to_node(eqn[1], options, varMap=varMap),
                    symbolic_to_node(eqn[2], options, varMap=varMap))
    elseif length(eqn) == 3
        return Node(op_idx,
                    symbolic_to_node(eqn[1], options, varMap=varMap),
                    _multiarg_split(op_idx, eqn[2:3], options, varMap))
    else
        # Minimize depth:
        split_point = round(Int, length(eqn) // 2)
        return Node(op_idx,
                    _multiarg_split(op_idx, eqn[1:split_point], options, varMap),
                    _multiarg_split(op_idx, eqn[split_point+1:end], options, varMap))
    end
end

# Equation:
function symbolic_to_node(eqn::T, options::Options;
                       varMap::Union{Array{String, 1}, Nothing}=nothing
                  )::Node where {T<:SymbolicUtils.Term{<:Number}}
    args = SymbolicUtils.arguments(eqn)
    l = symbolic_to_node(args[1], options, varMap=varMap)
    nargs = length(args)
    op = SymbolicUtils.operation(eqn)
    if nargs == 1
        op_idx = unaop_to_index(op, options)
        return Node(op_idx, l)
    else
        op_idx = binop_to_index(op, options)
        if nargs == 2
            r = symbolic_to_node(args[2], options, varMap=varMap)
            return Node(op_idx, l, r)
        else
            # TODO: Assert operator is +, *
            return _multiarg_split(op_idx, args, options, varMap)
        end
    end
end

function unaop_to_index(op::F, options::Options)::Int where {F<:SymbolicUtils.Sym}
    # In format _unaop1
    parse(Int, string(op.name)[7:end])
end

function binop_to_index(op::F, options::Options)::Int where {F<:SymbolicUtils.Sym}
    # In format _binop1
    parse(Int, string(op.name)[7:end])
end

function unaop_to_index(op::F, options::Options)::Int where {F<:Function}
    for i=1:options.nuna
        if op == options.unaops[i]
            return i
        end
    end
    error("Operator $(op) in simplified expression not found in options $(options.unaops)!")
end

function binop_to_index(op::F, options::Options)::Int where {F<:Function}
    for i=1:options.nbin
        if op == options.binops[i]
            return i
        end
    end
    error("Operator $(op) in simplified expression not found in options $(options.binops)!")
end

function varMap_to_index(var::Symbol, varMap::Array{String, 1})::Int
    str = string(var)
    for i=1:length(varMap)
        if str == varMap[i]
            return i
        end
    end
end

function varMap_to_index(var::Symbol, varMap::Nothing)::Int
    return parse(Int, string(var)[2:end])
end
