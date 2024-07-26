module OperatorEnumModule

abstract type AbstractOperatorEnum end

"""
    OperatorEnum

Defines an enum over operators, along with their derivatives.
# Fields
- `binops`: A tuple of binary operators. Scalar input type.
- `unaops`: A tuple of unary operators. Scalar input type.
"""
struct OperatorEnum{B,U} <: AbstractOperatorEnum
    binops::B
    unaops::U
end

"""
    GenericOperatorEnum

Defines an enum over operators, along with their derivatives.
# Fields
- `binops`: A tuple of binary operators.
- `unaops`: A tuple of unary operators.
"""
struct GenericOperatorEnum{B,U} <: AbstractOperatorEnum
    binops::B
    unaops::U
end

struct TensorOperator{Finplace,Fgrad,Fconstr,Fcompl}
    symbol_name::Symbol
        # the name of the operator
    # op::Fdirect
        # gets the value
    op!::Finplace
        # does the operation and puts the result into z
        # (l,[ r,] res) -> Nothing
        # inputs are arrays
    # op_gpu_config::Fgpuc
        # !!! return the gpu config (probably the number of threads)
        # ...
    # op_gpu::Fgpu
        # !!! perform the operation on the gpu, given the gpu config instance (probably the thread number) 
        # ....
    gradient!::Fgrad
        # computes the derivatives, given the derivative of the output (res) and stores it into ∂l 
        # (res, ∂res, l, ∂l [, r, ∂r], Val(comp)) -> Nothing
        # inputs are arrays
    push_constraints!::Fconstr 
        # appends the constraints when trying to infer the shape
        # ...
    # get_shape::Fshapef
        # gets the shape, given the (two) input shape(s)
        # (sl[, sr]) -> NTuple{N,Int64}
        # inputs are NTuple{N, Int64}
    complexity::Fcompl
        # obtains the amount of floating point operations the operation requires (given the shapes)
        # (sl[, sr]) -> Int64
        # inputs are NTuple{N, Int64}

    TensorOperator(; symbol_name, op!::F1, gradient!::F2, push_constraints!::F3, complexity::F4) where {F1,F2,F3,F4} = 
        new{F1,F2,F3,F4}(symbol_name, op!, gradient!, push_constraints!, complexity)
end

"""
    TensorOperatorEnum

Defines an enum of operators over tensors. The operators must be inplace.
"""
struct TensorOperatorEnum{
    NB, B <: NTuple{NB, TensorOperator},
    NU, U <: NTuple{NU, TensorOperator}
} <: AbstractOperatorEnum
    binops::B
    unaops::U

    TensorOperatorEnum{NB,B,NU,U}(b::B,u::U) where {NB,B,NU,U} = new{NB,B,NU,U}(b,u)
end

Base.copy(op::AbstractOperatorEnum) = op
# TODO: Is this safe? What if a vector is passed here?

end
