module OperatorEnumConstructionModule

using DispatchDoctor: @unstable

import ..OperatorEnumModule: AbstractOperatorEnum, OperatorEnum, TensorOperatorEnum, GenericOperatorEnum, TensorOperator
import ..NodeModule: Node, GraphNode, AbstractScalarExprNode, AbstractTensorExprNode, constructorof
import ..StringsModule: string_tree
import ..EvaluateModule: eval_tree_array, OPERATOR_LIMIT_BEFORE_SLOWDOWN
import ..EvaluateDerivativeModule: eval_grad_tree_array, _zygote_gradient
import ..EvaluationHelpersModule: _grad_evaluator

"""Used to set a default value for `operators` for ease of use."""
@enum AvailableOperatorTypes::UInt8 begin
    IsNothing
    IsOperatorEnum
    IsGenericOperatorEnum
end

# These constants are purely for convenience. Internal code
# should make use of `Node`, `string_tree`, `eval_tree_array`,
# and `eval_grad_tree_array` directly.

const LATEST_OPERATORS = Ref{Union{Nothing,AbstractOperatorEnum}}(nothing)
const LATEST_OPERATORS_TYPE = Ref{AvailableOperatorTypes}(IsNothing)
const LATEST_UNARY_OPERATOR_MAPPING = Dict{Function,fieldtype(Node{Float64}, :op)}()
const LATEST_BINARY_OPERATOR_MAPPING = Dict{Function,fieldtype(Node{Float64}, :op)}()
const ALREADY_DEFINED_UNARY_OPERATORS = (;
    operator_enum=Dict{DataType,Dict{Function,Bool}}(),
    generic_operator_enum=Dict{DataType,Dict{Function,Bool}}(),
)
const ALREADY_DEFINED_BINARY_OPERATORS = (;
    operator_enum=Dict{DataType,Dict{Function,Bool}}(),
    generic_operator_enum=Dict{DataType,Dict{Function,Bool}}(),
)
const LATEST_VARIABLE_NAMES = Ref{Vector{String}}(String[])
const LATEST_LOCK = Threads.SpinLock()

function Base.show(io::IO, tree::AbstractScalarExprNode)
    latest_operators_type = LATEST_OPERATORS_TYPE.x
    kwargs = (variable_names=LATEST_VARIABLE_NAMES.x,)
    if latest_operators_type == IsNothing
        return print(io, string_tree(tree; kwargs...))
    elseif latest_operators_type == IsOperatorEnum
        latest_operators = LATEST_OPERATORS.x::OperatorEnum
        return print(io, string_tree(tree, latest_operators; kwargs...))
    else
        latest_operators = LATEST_OPERATORS.x::GenericOperatorEnum
        return print(io, string_tree(tree, latest_operators; kwargs...))
    end
end
function Base.show(io::IO, tree::AbstractTensorExprNode)
    print(io, string_tree(tree, nothing))
end
@unstable function (tree::AbstractScalarExprNode)(X; kws...)
    Base.depwarn(
        "The `tree(X; kws...)` syntax is deprecated. Use `tree(X, operators; kws...)` instead.",
        :AbstractScalarExprNode,
    )
    latest_operators_type = LATEST_OPERATORS_TYPE.x

    latest_operators_type == IsNothing &&
        error("Please use the `tree(X, operators; kws...)` syntax instead.")

    if latest_operators_type == IsOperatorEnum
        latest_operators = LATEST_OPERATORS.x::OperatorEnum
        return tree(X, latest_operators; kws...)
    else
        latest_operators = LATEST_OPERATORS.x::GenericOperatorEnum
        return tree(X, latest_operators; kws...)
    end
end

@unstable function _grad_evaluator(tree::AbstractScalarExprNode, X; kws...)
    Base.depwarn(
        "The `tree'(X; kws...)` syntax is deprecated. Use `tree'(X, operators; kws...)` instead.",
        :AbstractScalarExprNode,
    )
    latest_operators_type = LATEST_OPERATORS_TYPE.x
    # return _grad_evaluator(tree, X, $operators; kws...)
    latest_operators_type == IsNothing &&
        error("Please use the `tree'(X, operators; kws...)` syntax instead.")
    latest_operators_type == IsGenericOperatorEnum &&
        error("Gradients are not implemented for `GenericOperatorEnum`.")

    latest_operators = LATEST_OPERATORS.x::OperatorEnum
    return _grad_evaluator(tree, X, latest_operators; kws...)
end

function set_default_variable_names!(variable_names::Vector{String})
    return LATEST_VARIABLE_NAMES.x = copy(variable_names)
end

Base.@deprecate create_evaluation_helpers!(operators) set_default_operators!(operators)

function set_default_operators!(operators::OperatorEnum)
    LATEST_OPERATORS.x = operators
    return LATEST_OPERATORS_TYPE.x = IsOperatorEnum
end
function set_default_operators!(operators::GenericOperatorEnum)
    LATEST_OPERATORS.x = operators
    return LATEST_OPERATORS_TYPE.x = IsGenericOperatorEnum
end

@unstable function lookup_op(@nospecialize(f), ::Val{degree}) where {degree}
    mapping = degree == 1 ? LATEST_UNARY_OPERATOR_MAPPING : LATEST_BINARY_OPERATOR_MAPPING
    if !haskey(mapping, f)
        error(
            "Convenience constructor for operator `$(f)` is out-of-date. " *
            "Please create an `OperatorEnum` (or `GenericOperatorEnum`) containing " *
            "the operator `$(f)` which will define the `$(f)` -> `Int` mapping.",
        )
    end
    return mapping[f]
end

@unstable function _unpack_broadcast_function(f)
    if f isa Broadcast.BroadcastFunction
        return Symbol(f.f), :(Broadcast.BroadcastFunction($(f.f)))
    else
        return Symbol(f), Symbol(f)
    end
end

function _validate_no_ambiguous_broadcasts(operators::AbstractOperatorEnum)
    for ops in (operators.binops, operators.unaops), op in ops
        if op isa Broadcast.BroadcastFunction &&
            (op.f in operators.binops || op.f in operators.unaops)
            throw(
                ArgumentError(
                    "Usage of both broadcasted and unbroadcasted operator `$(op.f)` is ambiguous",
                ),
            )
        end
    end
    return nothing
end

function empty_all_globals!(; force=true)
    if force || islocked(LATEST_LOCK)
        lock(LATEST_LOCK) do
            LATEST_OPERATORS.x = nothing
            LATEST_OPERATORS_TYPE.x = IsNothing
            empty!(LATEST_UNARY_OPERATOR_MAPPING)
            empty!(LATEST_BINARY_OPERATOR_MAPPING)
            LATEST_VARIABLE_NAMES.x = String[]
        end
    end
    return nothing
end

function _extend_unary_operator(
    f_inside::Symbol, f_outside::Symbol, type_requirements, internal
)
    quote
        @gensym _constructorof _AbstractScalarExprNode
        quote
            if $$internal
                import ..NodeModule.constructorof as $_constructorof
                import ..NodeModule.AbstractScalarExprNode as $_AbstractScalarExprNode
            else
                using DynamicExpressions:
                    constructorof as $_constructorof,
                    AbstractScalarExprNode as $_AbstractScalarExprNode
            end

            function $($f_outside)(
                l::N
            ) where {T<: $($type_requirements),N<: $_AbstractScalarExprNode{T}}
                return if (l.degree == 0 && l.constant)
                    $_constructorof(N)(T; val=$($f_inside)(l.val))
                else
                    latest_op_idx = $($lookup_op)($($f_inside), Val(1))
                    $_constructorof(N)(; op=latest_op_idx, l)
                end
            end
        end
    end
end

function _extend_binary_operator(
    f_inside::Symbol, f_outside::Symbol, type_requirements, build_converters, internal
)
    quote
        @gensym _constructorof _AbstractScalarExprNode
        quote
            if $$internal
                import ..NodeModule.constructorof as $_constructorof
                import ..NodeModule.AbstractScalarExprNode as $_AbstractScalarExprNode
            else
                using DynamicExpressions:
                    constructorof as $_constructorof,
                    AbstractScalarExprNode as $_AbstractScalarExprNode
            end

            function $($f_outside)(
                l::N, r::N
            ) where {T<: $($type_requirements),N<: $_AbstractScalarExprNode{T}}
                if (l.degree == 0 && l.constant && r.degree == 0 && r.constant)
                    $_constructorof(N)(T; val=$($f_inside)(l.val, r.val))
                else
                    latest_op_idx = $($lookup_op)($($f_inside), Val(2))
                    $_constructorof(N)(; op=latest_op_idx, l, r)
                end
            end
            function $($f_outside)(
                l::N, r::T
            ) where {T<: $($type_requirements),N<: $_AbstractScalarExprNode{T}}
                if l.degree == 0 && l.constant
                    $_constructorof(N)(T; val=$($f_inside)(l.val, r))
                else
                    latest_op_idx = $($lookup_op)($($f_inside), Val(2))
                    $_constructorof(N)(;
                        op=latest_op_idx, l, r=$_constructorof(N)(T; val=r)
                    )
                end
            end
            function $($f_outside)(
                l::T, r::N
            ) where {T<: $($type_requirements),N<: $_AbstractScalarExprNode{T}}
                if r.degree == 0 && r.constant
                    $_constructorof(N)(T; val=$($f_inside)(l, r.val))
                else
                    latest_op_idx = $($lookup_op)($($f_inside), Val(2))
                    $_constructorof(N)(;
                        op=latest_op_idx, l=$_constructorof(N)(T; val=l), r
                    )
                end
            end
            if $($build_converters)
                # Converters:
                function $($f_outside)(
                    l:: $_AbstractScalarExprNode{T1}, r:: $_AbstractScalarExprNode{T2}
                ) where {T1<: $($type_requirements),T2<: $($type_requirements)}
                    if l isa GraphNode || r isa GraphNode
                        error(
                            "Refusing to promote `GraphNode` as it would break the graph structure. " *
                            "Please convert to a common type first.",
                        )
                    end
                    return $($f_outside)(promote(l, r)...)
                end

                function $($f_outside)(
                    l:: $_AbstractScalarExprNode{T1}, r::T2
                ) where {T1<: $($type_requirements),T2<: $($type_requirements)}
                    return $($f_outside)(l, convert(T1, r))
                end
                function $($f_outside)(
                    l::T1, r:: $_AbstractScalarExprNode{T2}
                ) where {T1<: $($type_requirements),T2<: $($type_requirements)}
                    return $($f_outside)(convert(T2, l), r)
                end
            end
        end
    end
end


function _extend_binary_tensor_operator(fname, opnum, internal)
    return quote
        @gensym _constructorof _AbstractTensorExprNode
        quote
            if $$internal
                import ..NodeModule.constructorof as $_constructorof
                import ..NodeModule.AbstractTensorExprNode as $_AbstractTensorExprNode
            else
                using DynamicExpressions:
                    constructorof as $_constructorof,
                    AbstractTensorExprNode as $_AbstractTensorExprNode
            end

            function $($fname)(l::NodeT, r::NodeT) where {T,N,NodeT <: $_AbstractTensorExprNode{T,N}}
                $_constructorof(NodeT)(N,T; op=$($opnum), l=l, r=r)
            end
            function $($fname)(x:: $_AbstractTensorExprNode{T1,N1}, y:: $_AbstractTensorExprNode{T2,N2}) where {T1,N1,T2,N2}
                error("Incompatible types")
            end
        end
    end
end

function _extend_unary_tensor_operator(fname, opnum, internal)
    return quote
        @gensym _constructorof _AbstractTensorExprNode
        quote
            if $$internal
                import ..NodeModule.constructorof as $_constructorof
                import ..NodeModule.AbstractTensorExprNode as $_AbstractTensorExprNode
            else
                using DynamicExpressions:
                    constructorof as $_constructorof,
                    AbstractTensorExprNode as $_AbstractTensorExprNode
            end

            function $($fname)(l::NodeT) where {ElemType, NDims, NodeT <: $_AbstractTensorExprNode{ElemType,NDims}}
                $_constructorof(NodeT)(NDims,ElemType; op=$($opnum), l=l)
            end
        end
    end
end

# TODO: implement this correctly
# Now, it requires you to do using DynamicExpressions.OperatorEnumConstructionModule` before
function _extend_tensor_operators(operators, kws, __module__::Module)
    if !all(x -> first(x.args) ∈ (:empty_old_operators, :internal, :on_type), kws)
        error(
            "You passed the keywords $(kws), but only `empty_old_operators`, `internal`, `on_type` are supported.",
        )
    end

    internal_idx = findfirst(x -> hasproperty(x, :args) && first(x.args) == :internal, kws)
    internal = if internal_idx !== nothing
        @assert kws[internal_idx].head == :(=)
        kws[internal_idx].args[2]::Bool
    else
        false
    end

    @gensym fname opnum
    unaryx = _extend_unary_tensor_operator(fname, opnum, internal)
    binaryx = _extend_binary_tensor_operator(fname, opnum, internal)
    return quote
        # $(if internal
        #     :(using ..OperatorEnumConstructionModule: _extend_unary_tensor_operator as $unafunc)
        # else
        #     :(using DynamicExpressions.OperatorEnumConstructionModule: _extend_unary_operator as $unafunc)
        # end)
        lock($LATEST_LOCK) do
            for ($opnum, top) in enumerate($(operators).unaops)
                local $fname = top.symbol_name
                if isdefined(Base, $fname)
                    $fname = :(Base.$($fname))
                elseif isdefined($__module__, $fname)
                    $fname = :($($__module__).$($fname))
                end
                eval($unaryx)
            end
            for ($opnum, top) in enumerate($(operators).binops)
                local $fname = top.symbol_name
                if isdefined(Base, $fname)
                    $fname = :(Base.$($fname))
                elseif isdefined($__module__, $fname)
                    $fname = :($($__module__).$($fname))
                end
                eval($binaryx)
            end
        end
    end
end

function _extend_operators(operators, skip_user_operators, kws, __module__::Module)
    if !all(x -> first(x.args) ∈ (:empty_old_operators, :internal, :on_type), kws)
        error(
            "You passed the keywords $(kws), but only `empty_old_operators`, `internal`, `on_type` are supported.",
        )
    end

    empty_old_operators_idx = findfirst(
        x -> hasproperty(x, :args) && first(x.args) == :empty_old_operators, kws
    )
    internal_idx = findfirst(x -> hasproperty(x, :args) && first(x.args) == :internal, kws)
    on_type_idx = findfirst(x -> hasproperty(x, :args) && first(x.args) == :on_type, kws)

    empty_old_operators = if empty_old_operators_idx !== nothing
        @assert kws[empty_old_operators_idx].head == :(=)
        kws[empty_old_operators_idx].args[2]
    else
        true
    end

    on_type = if on_type_idx !== nothing
        @assert kws[on_type_idx].head == :(=)
        kws[on_type_idx].args[2]
    else
        nothing
    end

    internal = if internal_idx !== nothing
        @assert kws[internal_idx].head == :(=)
        kws[internal_idx].args[2]::Bool
    else
        false
    end

    @gensym f_inside f_outside skip type_requirements build_converters binary_exists unary_exists
    binary_ex = _extend_binary_operator(
        f_inside, f_outside, type_requirements, build_converters, internal
    )
    unary_ex = _extend_unary_operator(f_inside, f_outside, type_requirements, internal)
    #! format: off
    return quote
        local $type_requirements, $build_converters, $binary_exists, $unary_exists
        $(_validate_no_ambiguous_broadcasts)($operators)
        lock($LATEST_LOCK) do
        if isa($operators, $OperatorEnum)
            $type_requirements = $(on_type == nothing ? Number : on_type)
            $build_converters = $(on_type == nothing)
            if !haskey($(ALREADY_DEFINED_BINARY_OPERATORS).operator_enum, $type_requirements)
                $(ALREADY_DEFINED_BINARY_OPERATORS).operator_enum[$type_requirements] = Dict{Function,Bool}()
            end
            if !haskey($(ALREADY_DEFINED_UNARY_OPERATORS).operator_enum, $type_requirements)
                $(ALREADY_DEFINED_UNARY_OPERATORS).operator_enum[$type_requirements] = Dict{Function,Bool}()
            end
            $binary_exists = $(ALREADY_DEFINED_BINARY_OPERATORS).operator_enum[$type_requirements]
            $unary_exists = $(ALREADY_DEFINED_UNARY_OPERATORS).operator_enum[$type_requirements]
        else
            $type_requirements = $(on_type == nothing ? Any : on_type)
            $build_converters = false
            if !haskey($(ALREADY_DEFINED_BINARY_OPERATORS).generic_operator_enum, $type_requirements)
                $(ALREADY_DEFINED_BINARY_OPERATORS).generic_operator_enum[$type_requirements] = Dict{Function,Bool}()
            end
            if !haskey($(ALREADY_DEFINED_UNARY_OPERATORS).generic_operator_enum, $type_requirements)
                $(ALREADY_DEFINED_UNARY_OPERATORS).generic_operator_enum[$type_requirements] = Dict{Function,Bool}()
            end
            $binary_exists = $(ALREADY_DEFINED_BINARY_OPERATORS).generic_operator_enum[$type_requirements]
            $unary_exists = $(ALREADY_DEFINED_UNARY_OPERATORS).generic_operator_enum[$type_requirements]
        end
        if $(empty_old_operators)
            # Trigger errors if operators are not yet defined:
            empty!($(LATEST_BINARY_OPERATOR_MAPPING))
            empty!($(LATEST_UNARY_OPERATOR_MAPPING))
        end
        for (op, func) in enumerate($(operators).binops)
            local ($f_outside, $f_inside) = $(_unpack_broadcast_function)(func)
            local $skip = false
            if isdefined(Base, $f_outside)
                $f_outside = :(Base.$($f_outside))
            elseif $(skip_user_operators)
                $skip = true
            else
                $f_outside = :($($__module__).$($f_outside))
            end
            $(LATEST_BINARY_OPERATOR_MAPPING)[func] = op
            $skip && continue
            # Avoid redefining methods:
            if !haskey($unary_exists, func)
                eval($binary_ex)
                $(unary_exists)[func] = true
            end
        end
        for (op, func) in enumerate($(operators).unaops)
            local ($f_outside, $f_inside) = $(_unpack_broadcast_function)(func)
            local $skip = false
            if isdefined(Base, $f_outside)
                $f_outside = :(Base.$($f_outside))
            elseif $(skip_user_operators)
                $skip = true
            else
                $f_outside = :($($__module__).$($f_outside))
            end
            $(LATEST_UNARY_OPERATOR_MAPPING)[func] = op
            $skip && continue
            # Avoid redefining methods:
            if !haskey($binary_exists, func)
                eval($unary_ex)
                $(binary_exists)[func] = true
            end
        end
        end
    end
    #! format: on
end

"""
    @extend_operators operators [kws...]

Extends all operators defined in this operator enum to work on the
`Node` type. While by default this is already done for operators defined
in `Base` when you create an enum and pass `define_helper_functions=true`,
this does not apply to the user-defined operators. Thus, to do so, you must
apply this macro to the operator enum in the same module you have the operators
defined.
"""
macro extend_operators(operators, kws...)
    expected_type = AbstractOperatorEnum
    tensor_type = TensorOperatorEnum
    return esc(
        quote
            if !isa($(operators), $expected_type)
                error("You must pass an operator enum to `@extend_operators`.")
            elseif isa($(operators), $tensor_type)
                $(_extend_tensor_operators(operators, kws, __module__))
            else
                $(_extend_operators(operators, false, kws, __module__))
            end
        end,
    )
end

"""
    @extend_operators_base operators [kws...]

Similar to `@extend_operators`, but only extends operators already
defined in `Base`.
`kws` can include `empty_old_operators` which is default `true`,
and `internal` which is default `false`.
"""
macro extend_operators_base(operators, kws...)
    ex = _extend_operators(operators, true, kws, __module__)
    expected_type = AbstractOperatorEnum
    return esc(
        quote
            if !isa($(operators), $expected_type)
                error("You must pass an operator enum to `@extend_operators_base`.")
            end
            $ex
        end,
    )
end

"""
    OperatorEnum(; binary_operators=[], unary_operators=[],
                   define_helper_functions::Bool=true,
                   empty_old_operators::Bool=true)

Construct an `OperatorEnum` object, defining the possible expressions. This will also
redefine operators for `AbstractScalarExprNode` types, as well as `show`, `print`, and
`(::AbstractScalarExprNode)(X)`. It will automatically compute derivatives with `Zygote.jl`.

# Arguments
- `binary_operators::Vector{Function}`: A vector of functions, each of which is a binary
  operator.
- `unary_operators::Vector{Function}`: A vector of functions, each of which is a unary
  operator.
- `define_helper_functions::Bool=true`: Whether to define helper functions for creating
   and evaluating node types. Turn this off when doing precompilation. Note that these
   are *not* needed for the package to work; they are purely for convenience.
- `empty_old_operators::Bool=true`: Whether to clear the old operators.
"""
@unstable function OperatorEnum(;
    binary_operators=Function[],
    unary_operators=Function[],
    define_helper_functions::Bool=true,
    empty_old_operators::Bool=true,
    # Deprecated:
    enable_autodiff=nothing,
)
    enable_autodiff !== nothing && Base.depwarn(
        "The option `enable_autodiff` has been deprecated. " *
        "Differential operators are now automatically computed within the gradient call.",
        :OperatorEnum,
    )
    for (op, s) in ((binary_operators, "binary"), (unary_operators, "unary"))
        if length(op) > OPERATOR_LIMIT_BEFORE_SLOWDOWN
            @warn(
                "You have passed over $(OPERATOR_LIMIT_BEFORE_SLOWDOWN) $(s) operators. " *
                    "To prevent long compilation times, some optimizations will be disabled. " *
                    "If this presents an issue, please open an issue on https://github.com/SymbolicML/DynamicExpressions.jl"
            )
            break
        end
    end

    if define_helper_functions && any(
        op_set -> any(op -> op isa Broadcast.BroadcastFunction, op_set),
        (binary_operators, unary_operators),
    )
        # TODO: Fix issue with defining operators on a `BroadcastFunction`
        # and then on a regular function
        @warn "Using `BroadcastFunction` in an `OperatorEnum` is not yet stable"
    end

    operators = OperatorEnum(Tuple(binary_operators), Tuple(unary_operators))

    if define_helper_functions
        @extend_operators_base operators empty_old_operators = empty_old_operators
        set_default_operators!(operators)
    end

    return operators
end

"""
    GenericOperatorEnum(; binary_operators=[], unary_operators=[],
                          define_helper_functions::Bool=true, empty_old_operators::Bool=true)

Construct a `GenericOperatorEnum` object, defining possible expressions.
Unlike `OperatorEnum`, this enum one will work arbitrary operators and data types.
This will also redefine operators for `AbstractScalarExprNode` types, as well as `show`, `print`,
and `(::AbstractScalarExprNode)(X)`.

# Arguments
- `binary_operators::Vector{Function}`: A vector of functions, each of which is a binary
  operator.
- `unary_operators::Vector{Function}`: A vector of functions, each of which is a unary
  operator.
- `define_helper_functions::Bool=true`: Whether to define helper functions for creating
   and evaluating node types. Turn this off when doing precompilation. Note that these
   are *not* needed for the package to work; they are purely for convenience.
- `empty_old_operators::Bool=true`: Whether to clear the old operators.
"""
@unstable function GenericOperatorEnum(;
    binary_operators=Function[],
    unary_operators=Function[],
    define_helper_functions::Bool=true,
    empty_old_operators::Bool=true,
)
    @assert length(binary_operators) > 0 || length(unary_operators) > 0

    operators = GenericOperatorEnum(Tuple(binary_operators), Tuple(unary_operators))

    if define_helper_functions
        @extend_operators_base operators empty_old_operators = empty_old_operators
        set_default_operators!(operators)
    end

    return operators
end

# Predefine the most common operators so the errors
# are more informative
function _overload_common_operators()
    # Overload the operators in batches (so that we don't hit the warning
    # about too many operators)
    operators = OperatorEnum(
        (+, -, *, /, ^, max, min, mod),
        (sin, cos, tan, exp, log, log1p, log2, log10, sqrt, cbrt, abs, sinh),
    )
    @extend_operators(operators, empty_old_operators = false, internal = true)
    operators = OperatorEnum((), (cosh, tanh, atan, asinh, acosh, round, sign, floor, ceil))
    @extend_operators(operators, empty_old_operators = true, internal = true)

    empty!(LATEST_UNARY_OPERATOR_MAPPING)
    empty!(LATEST_BINARY_OPERATOR_MAPPING)
    return nothing
end
_overload_common_operators()

gradient(op, l) = (l,)
gradient(op, l, r) = (r, l)

broadcast_unaop(op::Fnum ; op_complexity=1, symbol::Union{Symbol, Nothing}=nothing) where {Fnum} = TensorOperator(;
    symbol_name = symbol === nothing ? Symbol(op) : symbol,
    op! = (l, res) -> (@. res = op(l)),
    # (l, res) -> (@. res = op(l)),
    gradient! = (res, ∂res, l, ∂l) -> begin
        map!(first, ∂l, gradient.(op, l))
        @. ∂l *= ∂res
    end,
    push_constraints! = push_constraints_broadcast,
    complexity = sl -> length(sl) * op_complexity
)

using ..ShapeInferenceModule: push_constraints_broadcast

broadcast_binop(op::Fnum ; op_complexity=1, symbol::Union{Symbol, Nothing}=nothing) where {Fnum} = TensorOperator(
    symbol_name = symbol === nothing ? Symbol(op) : symbol,
    op! = (l, r, res) -> (@. res = op(l, r)),
    # (l, r, res) -> (@. res = op(l, r)),
    gradient! = function(res, ∂res, l, ∂l, r, ∂r, ::Val{comp}) where {comp}
        grads = gradient.(op, l, r)
        if comp & 0b10 != 0
            sum!(∂l, ∂res .* map(x->x[1], grads))
        end
        if comp & 0b01 != 0
            sum!(∂r, ∂res .* map(x->x[2], grads))
        end
    end,
    push_constraints! = push_constraints_broadcast,
    complexity = (sl, sr) -> 
        prod(ntuple(i -> max(sl[i], sr[i]), Val(length(sl)))) * op_complexity
)

using ..OperatorEnumModule

# TODO: ~~move this to OperatorEnumConstruction~~ and make it similar to the other OperatorEnum
OperatorEnumModule.TensorOperatorEnum(; binary_operators::AbstractVector{<:TensorOperator}=[], unary_operators::AbstractVector{<:TensorOperator}=[]) = begin
    @assert length(binary_operators) + length(unary_operators) != 0
    TensorOperatorEnum{
        length(binary_operators), Tuple{map(typeof, binary_operators)...},
        length(unary_operators), Tuple{map(typeof, unary_operators)...},
    }(
        ntuple(i -> binary_operators[i], Val(length(binary_operators))),
        ntuple(i -> unary_operators[i], Val(length(unary_operators)))
    )
end

end
