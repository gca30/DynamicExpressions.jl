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

struct TensorOperator{Finplace,Fgrad,Fconstr,Fshapef,Fcompl}
    # op::Fdirect
        # gets the value
    op!::Finplace
        # does the operation and puts the result into z
        # (l,[ r,] res) -> Nothing
        # inputs are arrays
    extended_op!::Fext
        # does the operation, but with the first dimension being the datapoints
        # (l,[ r,] res) -> Noting
    # op_gpu_config::Fgpuc
        # !!! return the gpu config (probably the number of threads)
        # ...
    # op_gpu::Fgpu
        # !!! perform the operation on the gpu, given the gpu config instance (probably the thread number) 
        # ....
    gradient!::Fgrad
        # computes the derivatives, given the derivative of the output (res) and stores it into ∂l 
        # (res, ∂res, l, ∂l [, r, ∂r]) -> Nothing
        # inputs are arrays
    append_constraint!::Fconstr 
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
end

broadcast_unaop(op::Fnum; op_complexity=1) where {Fnum} = TensorOperator(
    (l, res) -> (@. res = op(l)),
    (l, res) -> (@. res = op(l)),
    (res, ∂res, l, ∂l) -> begin
        map!(first, ∂l, gradient.(op, l))
        @. ∂l *= ∂res 
    end,
    () -> error("Shape inference not yet defined"),
    sl -> length(sl) * op_complexity
)

broadcast_binop(op::Fnum; op_complexity=1) where {Fnum} = TensorOperator(
    (l, r, res) -> (@. res = op(l, r)),
    (l, r, res) -> (@. res = op(l, r)),
    (res, ∂res, l, ∂l, r, ∂r) -> begin
        grads = gradient.(op, l, r)
        sum!(∂l, ∂res .* map(x->x[1], grads))
        sum!(∂r, ∂res .* map(x->x[2], grads))
    end,
    () -> error("Shape inference not yet defined"),
    (sl::NTuple{N,Int64}, sr::NTuple{N,Int64}) where {N} -> 
        prod(ntuple(i -> max(sl[i], sr[i]), Val(N))) * op_complexity
)

"""
    TensorOperatorEnum

Defines an enum of operators over tensors. The operators must be inplace.
"""
struct TensorOperatorEnum{
    NB, B <: NTuple{NB,<:TensorOperator},
    NU, U <: NTuple{NU,<:TensorOperator}
} <: AbstractOperatorEnum
    binops::B
    unapos::U
end

Base.copy(op::AbstractOperatorEnum) = op
# TODO: Is this safe? What if a vector is passed here?

end
