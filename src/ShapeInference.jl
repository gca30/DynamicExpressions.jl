using DynamicExpressions: Node, AbstractNode, @extend_operators, OperatorEnum

c1 = Node(Array{Float64,4}; val=fill(1, (1, 1, 1, 1)))
c2 = Node(Array{Float64,4}; val=fill(2, (1, 1, 1, 1)))
c3 = Node(Array{Float64,4}; val=fill(3, (1, 1, 1, 1)))
c4 = Node(Array{Float64,4}; val=fill(4, (1, 1, 1, 1)))
c5 = Node(Array{Float64,4}; val=fill(5, (1, 1, 1, 1)))
x1 = Node(Array{Float64,4}; feature=1)
x2 = Node(Array{Float64,4}; feature=2)
x3 = Node(Array{Float64,4}; feature=3)

# dummy operators

function cross(x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T<:Number,N}
    @assert N >= 1
    @assert size(x) == size(y)
    @assert size(x, 1) == 3
    return rand(size(x))
end

function matmult(x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T<:Number,N}
    @assert N >= 2
    @assert size(x)[3:end] == size(y)[3:end]
    @assert size(x, 2) == size(y, 1)
    return rand(size(x, 1), size(x, 2), size(x)[3:end]...)
end

macro define_matmult(firstAxis, secondAxis, N, T)
    @assert typeof(firstAxis) == Int64
    @assert typeof(secondAxis) == Int64
    @assert typeof(N) == Int64
    @assert firstAxis != secondAxis
    if secondAxis < firstAxis
        firstAxis, secondAxis = secondAxis, firstAxis
    end
    funcname = symbol("mm" * string(firstAxis) * string(secondAxis))
    return quote
        function $(funcname)(x::AbstractArray{T,N}, y::AbstractArray{T,N})
            @assert size(x, $(secondAxis)) == size(y, $(firstAxis))
            @assert size(x)[1:($(firstAxis - 1))] == size(y)[1:($(firstAxis - 1))]
            @assert size(x)[($(firstAxis + 1)):($(secondAxis - 1))] ==
                size(y)[($(firstAxis + 1)):($(secondAxis - 1))]
            @assert size(x)[($(secondAxis + 1)):end] == size(y)[($(secondAxis + 1)):end]
            return dims = (
                size(x)[1:($(secondAxis - 1))]...,
                size(y, $(secondAxis)),
                size(x)[($(secondAxis + 1)):N]...,
            )
        end
    end
end

function transp(x::AbstractArray{T,N}) where {T<:Number,N}
    @assert N >= 2
    return rand(size(x, 2), size(x, 1), size(x)[3:end]...)
end

operators = OperatorEnum(;
    binary_operators=[.+, .*, cross, matmult], unary_operators=[transp]
)
@extend_operators(operators, on_type = Array{Float64,4})

trees = [
    matmult(transp(c4), cross(c1 + x1 * c2, x2 * c3)),
    matmult(matmult(c4, x3), (c1 + x1 * c2) * x2 * c3),
    matmult(c1, c2 + x2),
    matmult(c3, c4 + matmult(transp(matmult(c1, x1)), matmult(c2, c5 + transp(x2)))),
]

# Suppose we have a tree with return type Array{BT, N}, where BT = underlying number type, N = maximum allowed dimensions
# If at any point we have a lower dimension count than N, we set the size in the extra dimensions to 1
# Now we want to obtain the shapes of each node in the tree

# We traverse the tree and give each node a number
# We make a list of all the shape variables of the nodes, labeled a₁, a₂, ...
# When we encounter an operation, we impose a constraint on the shape variables
# For example, the mathematical expression of the constraint for the transpose 
#   nodes:      t2 = transpose(t1)
#   notation:   shape(t1) == (a1,a2), shape(t2) == (a3,a4)
#   constraint: (a1,a2,a3,a4) ∈ {(n,m,m,n) | n,m ∈ NN}
# More examples:
#   For the input we can impose a constraint of the type
#     (a₂₅,a₂₆,a₂₇,a₂₈) ∈ {(3,1,1,1)}
#   For a broadcasted sum (given by Julia Array rules)
#     (a₂₄,a₃₂,a₂₀) ∈ {(n,n,n)} ∪ {(n,1,n)} ∪ {(n,n,1)}
#   For a matrix multiplication:
#     (a₄₁,a₄₂,a₁,a₂,a₃₇,a₃₈) ∈ {(n,m,n,p,p,m)}

# Struct defintions

mutable struct ConstraintElem
    # if `isConstant` is set to true, the element will be a costnant of value `value`
    # otherwise, it will be a variable in the set, with the `value` being the index of the variable
    isConstant::Bool
    value::UInt64
end

@inline Base.zero(::Type{ConstraintElem}) = ConstraintElem(true, 0)
@inline function Base.zero(::Type{NTuple{M,T}}) where {M,T}
    return ntuple(Returns(zero(T)), Val(M))
end

struct Constraint{M,C}
    # the shape variables (aᵢ)
    indices::NTuple{M,UInt64}
    # Union of sets of tuples of constant/variable elements, which determine all possible allowed valuas of the shape variable tuple
    consSets::NTuple{C,NTuple{M,ConstraintElem}}
end
@inline Base.zero(::Type{Constraint{M,C}}) where {M,C} =
    Constraint{M,C}(zero(NTuple{M,Int64}), zero(NTuple{C,NTuple{M,ConstraintElem}}))

@inline function Base.convert(
    ::Type{Constraint{M2,C2}}, x::Constraint{M1,C1}
) where {M1,M2,C1,C2}
    return Constraint{M2,C2}(
        ntuple(i -> i > M1 ? 0 : x.indices[i], Val(M2)),
        ntuple(
            i -> if i > C1
                zero(NTuple{M2,ConstraintElem})
            else
                ntuple(j -> j > M1 ? zero(ConstraintElem) : x.consSets[i][j], Val(M2))
            end,
            C2,
        ),
    )
end

# Helper functions

# with tuples
@inline function removeNth(a::NTuple{N,T}, ix::UInt64)::NTuple{N,T} where {N,T}
    return setNth(a, ix, zero(T))
end
@inline function setNth(a::NTuple{N,T}, ix::UInt64, val::T)::NTuple{N,T} where {N,T}
    return ntuple(i -> i == ix ? val : a[i], Val(N))
end
@inline function swapNth(a::NTuple{N,T}, i1::Int64, i2::Int64)::NTuple{N,T} where {N,T}
    return ntuple(i -> i == i1 ? a[i2] : (i == i2 ? a[i1] : a[i]), Val(N))
end
@inline function removeNth(a::Constraint{M,C}, ix::Int64)::Constraint{M,C} where {M,C}
    return Constraint{M,C}(
        removeNth(a.indices, ix),
        map(set -> reset_max_var(removeNth(set, a_x_index)), r.consSets),
    )
end
function reorder(a::NTuple{N,T})::NTuple{N,T} where {N,T}
    final = a
    j = 1
    for i in 1:N
        (final[i] == zero(T)) && continue
        final = swapNth(final, i, j)
        j += 1
    end
    return final
end

# with ConstraintElem
@inline Base.:(==)(a::ConstraintElem, b::ConstraintElem) =
    a.isConstant == b.isConstant && a.value == b.value

# with type-stable Constraint
@inline effective_M(a::Constraint{M,C}) where {M,C} = count(!=(0), a.indices)
@inline is_nonempty_set(set) = any(ce -> ce.value != 0, set)
@inline effective_C(a::Constraint{M,C}) where {M,C} = count(is_nonempty_set, a.consSets)

@inline same_indices(a::Constraint{M,C}, b::Constraint{M,C}) where {M,C} =
    effective_M(a) == effective_M(b) && all(x -> x in a.indices, b.indices)

function reset_max_var(r::NTuple{M,ConstraintElem})::NTuple{M,ConstraintElem} where {M}
    covered = count(x -> x.isConstant, r)
    i = 1
    while covered < M
        mins = minimum(map(x -> (x.isConstant || x.value < i) ? M + 5 : x.value, r))
        new_count = count(x -> !x.isConstant && x.value == mins, r)
        covered += new_count
        if mins != i
            r = map(
                x -> (x.isConstant || x.value != mins) ? x : ConstraintElem(false, i), r
            )
        end
        i += 1
    end
    return r
end

function resolve_set_value(
    r::Constraint{M,C}, a_x::UInt64, value::UInt64
)::Constraint{M,C} where {M,C}
    !(a_x in r.indices) && return r
    a_x_index = UInt64(findfirst(x -> x == a_x, r.indices))
    is_single = count(x -> x != 0, r.indices) == 1
    if is_single
        has_the_value = any(
            set -> set[a_x_index].isConstant && set[a_x_index].value == value, r.consSets
        )
        if !has_the_value
            return convert(Constraint{M,C}, Constraint{1,0}((a_x,), Tuple{}()))
        end
    end
    return Constraint(
        removeNth(r.indices, a_x_index),
        map(
            set -> if set[a_x_index].isConstant
                set[a_x_index] == 0 ? set : removeNth(set, a_x_index)
            else
                varval = set[a_x_index].value
                reset_max_var(
                    removeNth(
                        map(ce -> if !ce.isConstant && ce.value == varval
                            ConstraintElem(true, value)
                        else
                            ce
                        end, set),
                        a_x_index,
                    ),
                )
            end,
            map(
                set -> if (set[a_x_index].isConstant && set[a_x_index].value != value)
                    zero(typeof(set))
                else
                    set
                end,
                r.consSets,
            ),
        ),
    )
end

# is A a subset of B ?
function Base.issubset(
    a::NTuple{M,ConstraintElem}, b::NTuple{M,ConstraintElem}
)::Bool where {M}
    constsMap = zero(NTuple{M,ConstraintElem})
    for i in eachindex(a)
        if b[i].isConstant
            if (a[i].isConstant && b[i].value != a[i].value) || !a[i].isConstant
                return false
            end
        else
            if constsMap[b[i].value].value == 0
                # if we don't know the value of 
                constsMap = setNth(constsMap, b[i].value, a[i])
            elseif !(constsMap[b[i].value] == a[i])
                return false
            end
        end
    end
    return true
end

function reorder(a::Constraint{M,C}) where {M,C}
    return Constraint{M,C}(reorder(a.indices), reorder(map(reorder, a.consSets)))
end

function intersection(a::Constraint{M,C}, b::Constraint{M,C})::Constraint{M,C} where {M,C}
    # a and b must have the same indices
    a = reorder(a)
    b = reorder(b)
    effM = effective_M(a)
    effCa = effective_C(a)
    effCb = effective_C(b)
    newBSets = map(
        sb -> begin
            sb == zero(typeof(sb)) && return sb
            if any(sa -> issubset(sb, sa) && !(sb == sa), a.consSets)
                sb
            else
                zero(typeof(sb))
            end
        end,
        b.consSets,
    )
    newASets = map(
        sa -> begin
            sa == zero(typeof(sa)) && return sa
            any(sb -> issubset(sa, sb), b.consSets) ? sa : zero(typeof(sa))
        end,
        a.consSets,
    )
    c2 = reorder(Constraint{M,2 * C}(a.indices, (newASets..., newBSets...)))
    @assert effective_C(c2) <= C
    return convert(Constraint{M,C}, c2)
end

# Printing constraints
using Unicode
function to_subscript_str(n)
    return map(
        k -> Unicode.julia_chartransform(
            Unicode.julia_chartransform('₁') + Int(k) - Int('1')
        ),
        string(n),
    )
end
function parensd(va, func, io)
    if length(va) == 1
        print(io, func(va[1]))
    else
        print(io, "(")
        for i in eachindex(va)
            print(io, func(va[i]))
            if i != length(va)
                print(io, ",")
            end
        end
        print(io, ")")
    end
end

function Base.show(io::IO, ce::ConstraintElem)
    if ce.value == 0
        print(io, "_")
    elseif ce.isConstant
        print(io, ce.value)
    else
        print(io, "_nmpqrtuvxyzbcdefghijka"[ce.value + 1])
    end
end

function Base.show(io::IO, c::Constraint)
    parensd(filter(x -> x != 0, c.indices), i -> "a" * to_subscript_str(i), io)
    print(io, " ∈ ")
    printed = false
    for i in eachindex(c.consSets)
        c.consSets[i] == zero(eltype(c.consSets)) && continue
        if printed
            print(io, " ∪ ")
        end
        print(io, "{")
        parensd(filter(ce -> ce.value != 0, c.consSets[i]), ce -> ce, io)
        print(io, "}")
        printed = true
    end
    if !printed
        print(io, "∅")
    end
end

function print_constraints(cs::Vector{Constraint{C,M}}) where {C,M}
    println("------- CONSTRAINTS --------")
    for i in eachindex(cs)
        count(!=(0), cs[i].indices) == 0 && continue
        println("R", i, ": ", cs[i])
    end
    return println()
end

function print_tree(nodes::AbstractVector{Node{AT}}) where {T,N,AT<:AbstractArray{T,N}}
    println("------- FLATTENED TREE --------")
    for i in eachindex(nodes)
        parensd((i - 1) * N .+ collect(1:N), ix -> "a" * to_subscript_str(ix), stdout)
        println(" -> ", nodes[i])
    end
    return println()
end

function print_final(
    final::AbstractVector{ConstraintElem}, nodes::AbstractVector{Node{AT}}
) where {N,BT,AT<:AbstractArray{BT,N}}
    println("------- SOLVED SO FAR --------")
    for i in eachindex(nodes)
        println(nodes[i], " -> ", ntuple(j -> final[(i - 1) * N + j], Val(N)))
    end
    return println()
end

function flatten_tree!(
    v::Vector{Node{AT}}, tree::Node{AT}
) where {BT,N,AT<:AbstractArray{BT,N}}
    if tree.degree == 0
        push!(v, tree)
    elseif tree.degree == 1
        flatten_tree!(v, tree.l)
        push!(v, tree)
    elseif tree.degree == 2
        flatten_tree!(v, tree.l)
        flatten_tree!(v, tree.r)
        push!(v, tree)
    end
end

MAX_M::Int64 = 1
MAX_C::Int64 = 1

# A push constraints macro to make constraints a lot more readable
macro push_constraint(cs, indices, sets...)
    (length(sets) == 0 || indices.head != :tuple) && throw("Wrong format of indices")
    N = length(indices.args)
    C = length(sets)
    global MAX_M
    MAX_M = max(N, MAX_M)
    global MAX_C
    MAX_C = max(C, MAX_C)

    for i in eachindex(sets)
        (
            sets[i].head != :tuple ||
            length(sets[i].args) != N ||
            !all(x -> typeof(x) == Int64 ? x >= 1 : typeof(x) == Symbol, sets[i].args)
        ) && throw("Wrong format of $(i)th set")
        d = Dict{Symbol,Int64}()
        for j in eachindex(sets[i].args)
            if typeof(sets[i].args[j]) == Int64
                sets[i].args[j] = ConstraintElem(true, sets[i].args[j])
            elseif haskey(d, sets[i].args[j])
                sets[i].args[j] = ConstraintElem(false, d[sets[i].args[j]])
            else
                d[sets[i].args[j]] = length(d) + 1
                sets[i].args[j] = ConstraintElem(false, d[sets[i].args[j]])
            end
        end
    end

    return quote
        push!(
            $(esc(cs)),
            convert(
                eltype($(esc(cs))),
                Constraint{$(N),$(C)}(
                    $(esc(indices)),
                    $(NTuple{C,NTuple{N,ConstraintElem}}(
                        map(set -> NTuple{N,ConstraintElem}(set.args), sets)
                    )),
                ),
            ),
        )
    end
end

function push_constraint_costants(
    cs::AbstractVector{Constraint{M,C}}, start_index, constants::NTuple{N,Int64}
) where {N,M,C}
    return push!(
        cs,
        convert(
            Constraint{M,C},
            Constraint{N,1}(
                ntuple(x -> (start_index - 1) * N + x, Val(N)),
                (map(x -> ConstraintElem(true, x), constants),),
            ),
        ),
    )
end

# Function that appends the constraints determined by the operator f onto the constraint vector cs
# N - max number of tensor dimensions
# indices - indices of the dimension variables involved in the operations
#   indices[1]+1, indices[1]+2, ..., indices[1]+N represent the dimensions of the output node
#   indices[2]+1, indices[2]+2, ..., indices[2]+N represent the dimensions of the left node
#   indices[3]+1, indices[3]+2, ..., indices[3]+N represent the dimensions of the right node (if applicable)
# Unary operators have 2-tuple, while binary ops have a 3-tuple as argument into the function
function append_constraints!(
    f::F, cs::Vector{Constraint{M,C}}, N::Int64, indices::Tuple{Int64,Int64}
) where {F,M,C}
    return error(
        "Unimplemented append_constraints! function for unary operator $(f)"
    )
end
function append_constraints!(
    f::F, cs::Vector{Constraint{M,C}}, N::Int64, indices::Tuple{Int64,Int64,Int64}
) where {F,M,C}
    return error(
        "Unimplemented append_constraints! function for binary operator $(f)"
    )
end

# Implementations of append_constraints!

append_constraints_broadcasting(cs::Vector{Constraint{M,C}}, (p, l, r), ::Val{N}) where {M,C,N} = begin
    for i in 1:N
        @push_constraint(cs, (p + i, l + i, r + i), (n, n, n), (n, 1, n), (n, n, 1))
    end
end

append_constraints_binary_all_equal(cs::Vector{Constraint{M,C}}, (p,l,r), ::Val{N}) where {N,M,C} = begin
    for i in 1:N
        @push_constraint(cs, (p + i, l + i, r + i), (n, n, n))
    end
end

append_constraints_unary_all_equal(cs::Vector{Constraint{M,C}}, (p,l), ::Val{N}) where {N,M,C} = begin
    for i in 1:N
        @push_constraint(cs, (p + i, l + i), (n, n))
    end
end

function append_constraints!(
    ::typeof(.+), cs2::Vector{Constraint{M,C}}, N::Int64, indices::Tuple{Int64,Int64,Int64}
) where {M,C}
    p, l, r = indices
    for i in 1:N
        @push_constraint(cs2, (p + i, l + i, r + i), (n, n, n), (n, 1, n), (n, n, 1))
    end
end

function append_constraints!(
    ::typeof(.*), cs1::Vector{Constraint{M,C}}, N::Int64, indices::Tuple{Int64,Int64,Int64}
) where {M,C}
    p, l, r = indices
    for i in 1:N
        @push_constraint(cs1, (p + i, l + i, r + i), (n, n, n), (n, 1, n), (n, n, 1))
    end
end

function append_constraints!(
    ::typeof(transp), cs::Vector{Constraint{M,C}}, N::Int64, indices::Tuple{Int64,Int64}
) where {M,C}
    p, c = indices
    @push_constraint(cs, (p + 1, p + 2, c + 1, c + 2), (n, m, m, n))
    for i in 3:N
        @push_constraint(cs, (p + i, c + i), (n, n))
    end
end

function append_constraints!(
    ::typeof(matmult),
    cs::Vector{Constraint{M,C}},
    N::Int64,
    indices::Tuple{Int64,Int64,Int64},
) where {M,C}
    p, l, r = indices
    @push_constraint(cs, (p + 1, p + 2, l + 1, l + 2, r + 1, r + 2), (n, m, n, p, p, m))
    for i in 3:N
        @push_constraint(cs, (p + i, l + i, r + i), (n, n, n), (n, 1, n), (n, n, 1))
    end
end

function append_constraints!(
    ::typeof(cross),
    cs::Vector{Constraint{M,C}},
    N::Int64,
    indices::Tuple{Int64,Int64,Int64},
) where {M,C}
    p, l, r = indices
    @push_constraint(cs, (p + 1, l + 1, r + 1), (3, 3, 3))
    for i in 2:N
        @push_constraint(cs, (p + i, l + i, r + i), (n, n, n), (n, 1, n), (n, n, 1))
    end
end

function inside_substitution(
    c::Constraint{M,C},
    cs::Vector{Constraint{M,C}},
    nodes::Vector{Node{AT}},
    solved::Dict{Node{AT},NTuple{N,Int64}}
)::Bool where {M,C,AT,N}
    effM, effC = effective_M(c), effective_C(c)
    !(effM == 1 && effC == 1) && return false
    a_x_index = findfirst(!=(0), c.indices)
    set_index = findfirst(s -> s[a_x_index].value != 0, c.consSets)
    !(c.consSets[set_index][a_x_index].isConstant) && return false
    @show a_x_index
    @show set_index
    a_x = c.indices[a_x_index]
    value = c.consSets[set_index][a_x_index].value
    node = nodes[div(a_x - 1, N) + 1]
    #=D=# # println("Applying substitution ", c)    
    if solved[node][mod(a_x - 1, N) + 1] != 0
        #=D=# # println(map(node -> solved[node], nodes))        
        error("Already set " * string(solved[node]))
    end
    map!(cst -> resolve_set_value(cst, a_x, value), cs, cs)
    #=D=# # print_constraints(cs)    
    solved[node] = setNth(solved[node], UInt64(mod(a_x - 1, N) + 1), Int64(value))
    return true
end

function splitting(
    c::Constraint{M,C}, cs::Vector{Constraint{M,C}}, ci::Int64
)::Bool where {M,C}
    effM, effC = effective_M(c), effective_C(c)
    (effM == 0 || effC == 0 || effM == 1) && return false
    first = findfirst(is_nonempty_set, c.consSets)
    mask::NTuple{M,UInt64} = map(ce -> ce.isConstant ? ce.value : 0, c.consSets[first])
    for si in (first + 1):C
        for cei in 1:M
            !any(ce -> ce.value != 0, c.consSets[si]) && continue
            if (
                !c.consSets[si][cei].isConstant || mask[cei] != c.consSets[si][cei].value
            ) && mask[cei] != 0
                mask = setNth(mask, UInt64(cei), UInt64(0))
            end
        end
    end
    sum(mask) == 0 && return false
    #=D=# # println("Applying splitting for ", c)    
    for ix in 1:M
        if mask[ix] != 0
            push!(
                cs,
                convert(
                    Constraint{M,C},
                    Constraint{1,1}((c.indices[ix],), ((ConstraintElem(true, mask[ix]),),)),
                ),
            )
        end
    end
    cs[ci] = Constraint{M,C}(
        ntuple(i -> mask[i] != 0 ? 0 : c.indices[i], Val(M)),
        map(
            set -> ntuple(i -> mask[i] != 0 ? zero(ConstraintElem) : set[i], Val(M)),
            c.consSets,
        ),
    )
    #=D=# # print_constraints(cs)    
    return true
end

function simplification(
    c::Constraint{M,C}, cs::Vector{Constraint{M,C}}, ci::Int64
)::Bool where {M,C}
    effM, effC = effective_M(c), effective_C(c)
    (effM != 1 || effC == 1 || effC == 0) && return false
    mask = ntuple(
        si -> begin
            for pi in eachindex(c.consSets)
                pi == si && continue
                # if issubset(c.consSets[si], c.consSets[pi]) && (pi < si ? true : !(c.consSets[si] == c.consSets[pi]))
                if (pi < si ? false : (c.consSets[si] == c.consSets[pi]))
                    return 1 # delete
                end
            end
            return 0 # keep
        end, Val(C)
    )
    sum(mask) == 0 && return false
    #=D=# # println("Applying simplification on ", cs[ci])    
    cs[ci] = Constraint{M,C}(
        c.indices,
        ntuple(i -> mask[i] == 0 ? c.consSets[i] : zero(eltype(c.consSets)), Val(C)),
    )
    #=D=# # print_constraints(cs)    
    return true
end

function outside_substitution(
    final::AbstractVector{ConstraintElem}, constraint::Constraint{M,C}
) where {M,C}
    next_v = maximum(ce -> ce.isConstant ? 0 : ce.value, final) + 1
    set_index = findfirst(set -> any(ce -> ce.value != 0, set), constraint.consSets)
    constsMap = ntuple(Returns(zero(ConstraintElem)), Val(M))
    for i in 1:M
        constraint.indices[i] == 0 && continue
        a_x = constraint.indices[i]
        is_const = constraint.consSets[set_index][i].isConstant
        value = constraint.consSets[set_index][i].value
        if final[a_x].value == 0
            if is_const
                final[a_x] = ConstraintElem(true, value)
            else
                if constsMap[value].value == 0
                    final[a_x] = ConstraintElem(false, next_v)
                    constsMap = setNth(constsMap, value, final[a_x])
                    next_v += 1
                else
                    final[a_x] = constsMap[value]
                end
            end
        elseif final[a_x].isConstant
            if is_const
                value == final[a_x].value ? continue : return a_x, value
            else
                if constsMap[value].value == 0
                    constsMap = setNth(constsMap, value, final[a_x])
                elseif constsMap[value].isConstant
                    if constsMap[value].value == final[a_x].value
                        continue
                    else
                        return a_x, constsMap[value].value
                    end
                else
                    varval = constsMap[value].value
                    map!(
                        ce -> (!ce.isConstant && ce.value == varval) ? final[a_x] : ce,
                        final,
                        final,
                    )
                    constsMap = setNth(constsMap, value, final[a_x])
                end
            end
        else
            if is_const
                varval = final[a_x].value
                map!(
                    ce -> if (!ce.isConstant && ce.value == varval)
                        ConstraintElem(true, value)
                    else
                        ce
                    end,
                    final,
                    final,
                )
            else
                if constsMap[value].value == 0
                    constsMap = setNth(constsMap, value, final[a_x])
                elseif constsMap[value].isConstant
                    varval = final[a_x].value
                    map!(
                        ce -> if (!ce.isConstant && ce.value == varval)
                            ConstraintElem(true, constsMap[value].value)
                        else
                            ce
                        end,
                        final,
                        final,
                    )
                else
                    if constsMap[value].value == final[a_x].value
                        continue
                    else
                        varval = final[a_x].value
                        map!(
                            ce -> if (!ce.isConstant && ce.value == varval)
                                constsMap[value]
                            else
                                ce
                            end,
                            final,
                            final,
                        )
                    end
                end
            end
        end
    end
    return 0, 0
end

"""
    inference_iteration(cs::AbstractVector{Constraint{M,C}}, final::AbstractVector{ConstraintElem))::Bool where {M,C}

Does one iteration of the shape inference process, defined as:

    loop until no changes:
        substitute single choice constraints into the final array
        substitute constants from final array into the constraints vector
        split independent terms in constraints
        simplify constraints to simplest form
    unify constraints with the same input condition

"""
function inference_iteration(
    cs::AbstractVector{Constraint{M,C}},
    final::AbstractVector{ConstraintElem},
    nodes::AbstractVector{Node{AT}},
) where {M,C,N,BT,AT<:AbstractArray{BT,N}}
    should_continue = true
    iterations = 0
    while should_continue
        should_continue = false
        iterations += 1

        # Outisde substitution
        for ci in eachindex(cs)
            effective_C(cs[ci]) != 1 && continue
            should_continue = true
            #=D=# # println("Applying ", cs[ci])            
            a_x, nv = outside_substitution(final, cs[ci])
            if a_x != 0
                dimi = mod(a_x - 1, N) + 1
                ni = div(a_x - 1, N) + 1
                error(
                    "Node $(nodes[ni]) in dimension $(dimi) cannot have sizes $(final[a_x]) and $(nv)",
                )
            end
            cs[ci] = zero(eltype(cs))
            #=D=# # println(final)            
            #=D=# # print_final(final, nodes)            
        end

        # Inside Substitution
        for ai in eachindex(final)
            (final[ai].value == 0 || !final[ai].isConstant) && continue
            map!(c -> resolve_set_value(c, UInt64(ai), final[ai].value), cs, cs)
        end

        # Splitting
        for ci in eachindex(cs)
            should_continue |= splitting(cs[ci], cs, ci)
        end

        # Simplification
        for ci in eachindex(cs)
            should_continue |= simplification(cs[ci], cs, ci)
        end

        # Remove illegal or redundant constraints
        filter!(c -> effective_M(c) != 0, cs)
        if count(c -> effective_C(c) == 0, cs) != 0
            #=D=# # print_constraints(cs)            
            i = findfirst(c -> effective_C(c) == 0, cs)
            a_x_index = findfirst(!=(0), cs[i].indices)
            a_x = cs[i].indices[a_x_index]
            #=D=# # print_final(final, nodes)            
            return error(
                "Could not find a potential size for node $(nodes[div(a_x-1, N)+1]) in dimension $(mod(a_x-1,N)+1)",
            )
        end
    end

    # Union
    for ci in eachindex(cs)
        for cj in 1:(ci - 1)
            effective_M(cs[ci]) == 0 && continue
            !same_indices(cs[ci], cs[cj]) && continue
            should_continue = true
            #=D=# # println("Applying union on ", cs[cj], " and ", cs[ci])            
            cs[cj] = intersection(cs[ci], cs[cj])
            cs[ci] = zero(Constraint{M,C})
            #=D=# # print_constraints(cs)            
        end
    end

    filter!(c -> effective_M(c) != 0, cs)
    if count(c -> effective_C(c) == 0, cs) != 0
        index = findfirst(c -> effective_C(c) == 0, cs)
        a_x = cs[index].indices[findfirst(!=(0), cs[index].indices)]
        return error(
            "Could not find a potential size for node $(nodes[div(a_x-1, N)+1]) in dimension $(mod(a_x-1,N)+1)",
        )
    end

    return !(iterations == 1)
end

"""
    _shape_inference(nodes::Vector{Node{AT}}, operators::O, ::Val{M}, ::Val{C}, featureSizes::NTuple{K,NTuple{N,Int64}})::Dict{Node{AT}, NTuple{N, Int64}} where {M,C,BT,N,K,O<:OperatorEnum,AT<:AbstractArray{BT,N}}

Internal implementation of shape_inference. Its pseudocode looks like this:

    traverse the tree:
        for features, append their shape as constraints
        for non-terminals, append the constraints of their operator
    loop until no constraints:
        loop until no changes:
            do a iteration
            if a constraint has multiple branches, choose a random one
        if the final answer still contains variables, choose randomly
    
"""
function _shape_inference(
    nodes::Vector{Node{AT}},
    operators::O,
    ::Val{M},
    ::Val{C},
    featureSizes::NTuple{K,NTuple{N,Int64}},
) where {M,C,BT,N,K,O<:OperatorEnum,AT<:AbstractArray{BT,N}}

    # TODO: the dict fucks up if you have 2 constant nodes with the same value
    # this could be solved by searching linearly in the array in reverse order starting from the parent
    dict = Dict{Node{AT},Int64}()
    solved = Dict{Node{AT},NTuple{N,Int64}}()
    final = fill(zero(ConstraintElem), N * length(nodes))
    for i in eachindex(nodes)
        dict[nodes[i]] = i
    end

    # Adding all the constraints
    cs = Vector{Constraint{M,C}}()
    sizehint!(cs, length(nodes) * N)
    for i in eachindex(nodes)
        if i == length(nodes)
            push_constraint_costants(cs, i, featureSizes[end])
        end
        if nodes[i].degree == 0 && !nodes[i].constant
            push_constraint_costants(cs, i, featureSizes[nodes[i].feature])
        elseif nodes[i].degree == 1
            append_constraints!(
                operators.unaops[nodes[i].op],
                cs,
                N,
                ((i - 1) * N, (dict[nodes[i].l] - 1) * N),
            )
        elseif nodes[i].degree == 2
            append_constraints!(
                operators.binops[nodes[i].op],
                cs,
                N,
                ((i - 1) * N, (dict[nodes[i].l] - 1) * N, (dict[nodes[i].r] - 1) * N),
            )
        end
    end

         # print_constraints(cs)    

    while length(cs) != 0
        should_continue = true
        while should_continue
            should_continue = false

            should_continue |= inference_iteration(cs, final, nodes)
            #=D=# # println(final)            
            #=D=# # print_constraints(cs)            

            # Choice
            independents = 0
            for ci in eachindex(cs)
                (effective_M(cs[ci]) == 0) && continue
                dependent = any(
                    cj -> if cj == ci
                        false
                    else
                        any(a_x -> a_x == 0 ? false : (a_x in cs[cj].indices), cs[ci].indices)
                    end,
                    eachindex(cs),
                )
                dependent && continue
                independents += 1
                #=D=# # println("Chose ", cs[ci])                
                #=D=# # print_constraints(cs)                
                effC = effective_C(cs[ci])
                C_index = trunc(Int64, rand() * effC) + 1
                cs[ci] = reorder(cs[ci])
                set = cs[ci].consSets[C_index]
                a_x, nv = outside_substitution(
                    final, convert(Constraint{M,C}, Constraint{M,1}(cs[ci].indices, (set,)))
                )
                if a_x != 0
                    dimi = mod(a_x - 1, N) + 1
                    ni = div(a_x - 1, N) + 1
                    error(
                        "Node $(nodes[ni]) in dimension $(dimi) cannot have sizes $(final[a_x]) and $(nv)",
                    )
                end
                cs[ci] = zero(Constraint{M,C})
                should_continue = true
                break
            end

            if independents == 0 && length(cs) != 0
                ci = abs(rand(Int64)) % length(cs) + 1
                effC = effective_C(cs[ci])
                C_index = trunc(Int64, rand() * effC) + 1
                cs[ci] = reorder(cs[ci])
                set = cs[ci].consSets[C_index]
                a_x, nv = outside_substitution(
                    final, convert(Constraint{M,C}, Constraint{M,1}(cs[ci].indices, (set,)))
                )
                if a_x != 0
                    dimi = mod(a_x - 1, N) + 1
                    ni = div(a_x - 1, N) + 1
                    error(
                        "Node $(nodes[ni]) in dimension $(dimi) cannot have sizes $(final[a_x]) and $(nv)",
                    )
                end
                cs[ci] = zero(Constraint{M,C})
                should_continue = true
            end
        end

        # Replacing variables
        non_ones = count(ce -> ce.isConstant && ce.value != 1, final)
        avgdim = div(sum(ce -> ce.isConstant ? ce.value - 1 : 0, final), non_ones) + 1
        for i in eachindex(final)
            final[i].isConstant && continue
            any(
                c -> any(
                    a_x ->
                        a_x != 0 &&
                            !final[a_x].isConstant &&
                            final[a_x].value == final[i].value,
                    c.indices,
                ),
                cs,
            ) && continue
            gendim = abs(rand(Int64)) % avgdim + div(avgdim, 2)
            push!(
                cs,
                convert(
                    Constraint{M,C},
                    Constraint{1,1}((i,), ((ConstraintElem(true, gendim),),)),
                ),
            )
        end
    end

    #=D=# # println("Finished")    
    #=D=# # print_constraints(cs)    
    #=D=# # println(solved)    

    for ce in final
        if !ce.isConstant || ce.value == 0
            error("Error")
        end
    end
    for i in eachindex(nodes)
        solved[nodes[i]] = ntuple(j -> final[(i - 1) * N + j].value, Val(N))
    end
    return solved
end

"""
    shape_inference(tree::Node{AT}, operators::O, featureSizes::NTuple{K,NTuple{N,Int64}}; throw_errors::Val{errors}=Val(false))
        ::Tuple{Dict{Node{AT}, NTuple{N, Int64}}, Bool} where {errors,BT,N,K,O<:OperatorEnum,AT<:AbstractArray{BT,N}}

Infers the required shapes of every term in a tree, given the tree, the operation, 
`featureSizes` (which contains the shapes of the inputs, with the shape of the output
appended at the end). It returns a dictionary from each node in the tree to its shape, and whether
the expression is shape-consistent.
If `throw_errors` is turned off, it will return an empty dictionary and false if it fails.
"""
function shape_inference(
    tree::Node{AT},
    operators::O,
    featureSizes::NTuple{K,NTuple{N,Int64}};
    throw_errors::Val{errors}=Val(false),
) where {errors,BT,N,K,O<:OperatorEnum,AT<:AbstractArray{BT,N}}

    # Flatten tree and create useful dictionaries
    M = max(MAX_M, N)
    C = MAX_C

    nodes = Node{AT}[]
    flatten_tree!(nodes, tree)
    if errors
        return _shape_inference(nodes, operators, Val(M), Val(C), featureSizes), true
    else
        return try
            _shape_inference(nodes, operators, Val(M), Val(C), featureSizes), true
        catch e
            Dict{Node{AT},NTuple{N,Int64}}(), false
        end
    end
end
