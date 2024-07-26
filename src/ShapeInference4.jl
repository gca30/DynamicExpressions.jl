
module ShapeInferenceModule

using ...NodeModule: TensorNode
using ..OperatorEnumModule: TensorOperatorEnum
using ..FlattenedTensorListModule: FlattenedTensorList
using ...NodeUtilsModule: renumber_nodes!, number_of_indices

# ----------------------
# DEFINITIONS
# ----------------------

mutable struct Constraint
    tuple :: Vector{Int32} # the a-variable numbers
    sets :: Array{Int32, 3} 
    # the axes represent:
        # 1. the union axis (how many options are there)
        # 2. the a-variable axis (wth length equal to length(tuple))
        # 3. the m-variable axis (with the first element in that axis being constant and others variables)
end

# the outside is stored as:
mutable struct CombinedConstraints
    values :: Matrix{Int32}
    # the union axis size is implied to be 1, and the tuple is taken to be the range from 1 to the number of avars
    CombinedConstraints(A, M) = new(zeros(Int32, A, M))
end



# ----------------------
# HELPER FUNCTIONS
# ----------------------

@inline Base.zero(::Type{Constraint}) = Constraint(Vector{Int32}(undef, 0), Array{Int32, 3}(undef, 0, 0, 0))
@inline sets_size(c::Constraint) = size(c.sets, 1)
@inline avars_size(c::Constraint) = size(c.sets, 2)
@inline mvars_size(c::Constraint) = size(c.sets, 3)

@inline sets_size(cb::CombinedConstraints) = 1
@inline avars_size(cb::CombinedConstraints) = size(cb.values, 1)
@inline mvars_size(cb::CombinedConstraints) = size(cb.values, 2)

@inline expand_mvars(mat::Matrix) = cat(mat, zeros(eltype(mat), size(mat)...); dims=2)
@inline expand_mvars!(cb::CombinedConstraints) = begin
    cb.values = expand_mvars(cb.values)
end

@inline mvar_occupied(cb::CombinedConstraints, mx) = any(!=(0), @view(cb.values[:,mx]))
@inline mvar_occupied(cbv::AbstractMatrix, mx) = any(!=(0), @view(cbv[:,mx]))
@inline avar_occupied(cb::CombinedConstraints, ax) = any(!=(0), @view(cb.values[ax,:]))
@inline avar_constant(cb::CombinedConstraints, ax) = cb.values[ax,1] != 0 && all(==(0), @view(cb.values[ax,2:end]))
@inline avar_count(cb::CombinedConstraints) = count(ax -> avar_occupied(cb, ax), axes(cb.values, 1))
@inline mvar_count(cb::CombinedConstraints) = count(mx -> mvar_occupied(cb, mx), 2:size(cb, 2))+1

first_unused_var!(cb::CombinedConstraints) = begin
    for mx in axes(cb.values, 2)
        if mx == 1 continue end
        if !mvar_occupied(cb, mx) return mx end
    end
    mx = size(cb.values, 2) + 1
    expand_mvars!(cb)
    return mx
end
first_unused_var!(cb::CombinedConstraints, constsMap::CombinedConstraints) = begin
    for mx in axes(cb.values, 2)
        if mx == 1 continue end
        if !mvar_occupied(cb, mx) && !mvar_occupied(constsMap, mx) return mx end
    end
    mx = size(cb.values, 2) + 1
    expand_mvars!(cb)
    expand_mvars!(constsMap)
    return mx
end

function replace_var!(cb::CombinedConstraints, mx, val::AbstractVector{Int32})
    @assert size(cb.values, 2) == length(val)
    for ax in axes(cb.values, 1)
        if cb.values[ax, mx] == 0 
            continue
        end
        # print("finally found m$(mx) in: a$(ax) = ")
        # print_mvars(stdout, cb.values[ax,:])
        # print("\n")
        coef = cb.values[ax, mx]
        cb.values[ax, mx] = 0
        @. @view(cb.values[ax,:]) += val * coef
    end
end

function renormalize_mvars!(cbv::AbstractMatrix{Int32})
    nmx = 1
    omx = 1
    while omx < size(cbv, 2)
        omx+=1
        if mvar_occupied(cbv, omx)
            nmx += 1
            if nmx == omx continue end
            @. @view(cbv[:,nmx]) = @view(cbv[:, omx])
        end
    end 
    @. @view(cbv[:,(nmx+1):end]) = 0
    return nmx
end

function renormalize_mvars!(cb::CombinedConstraints)
    nmx = 1
    omx = 1
    while omx < size(cb.values, 2)
        omx+=1
        if mvar_occupied(cb.values, omx)
            nmx += 1
            if nmx == omx continue end
            @. @view(cb.values[:,nmx]) = @view(cb.values[:, omx])
        end
    end 
    cb.values = copy(@view(cb.values[:, 1:nmx]))
    return nmx
end

function check_valid(cb::CombinedConstraints)
    for ax in axes(cb.values, 1)
        if any(!=(0), @view(cb.values[ax,2:end])) continue end
        if cb.values[ax,1] < 0
            return 3, ax
        end
    end
    return 0, 0
end



# ----------------------
# ALGORITHM IMPLEMENTATION
# ----------------------

function solve_dioph(c::Constraint, cb::CombinedConstraints, constsMap::CombinedConstraints, axc::Integer)
    error("solve_dioph not yet implemented")
    return 1000, 1
end

function outsubst(c::Constraint, cb::CombinedConstraints)
    Ac = avars_size(c)
    Mc = mvars_size(c)
    Mcb = mvars_size(cb)
    constsMap = CombinedConstraints(Mc, Mcb)
    # the avars of constsMap are mvars in c
    #   with the first one being a temporary array
    # the mvars of constsMap are mvars in cb

    for axc in 1:Ac
        avc = c.tuple[axc]
        axcb = avc
        if !avar_occupied(cb, axcb)
            # we haven't encountered this a-var in the combined constants so far
            cb.values[avc, 1] = c.sets[1, axc, 1]
            for mxc in 2:Mc
                if c.sets[1, axc, mxc] == 0 continue end
                if !avar_occupied(constsMap, mxc)
                    # if we don't know the value of a m-var in c, we say it is a new m-var in cb
                    mxcb = first_unused_var!(cb, constsMap)
                    constsMap.values[mxc, mxcb] = 1
                end
                # println(@view(constsMap.values[mxc, :]) .* c.sets[1, axc, mxc])
                @. @view(cb.values[axcb, :]) += @view(constsMap.values[mxc, :]) * c.sets[1, axc, mxc]
            end
            continue
        end
        # we have encountered this a-var in the combined constants already

        # we require all m-vars in c to have a corresponding value in cb
        for mxc in 2:Mc
            if c.sets[1, axc, mxc] == 0 continue end
            if !avar_occupied(constsMap, mxc)
                # if we don't know the value of a m-var in c, we say it is a new m-var in cb
                mxcb = first_unused_var!(cb, constsMap)
                constsMap.values[mxc, mxcb] = 1
            end
        end

        @. @view(constsMap.values[1, :]) = - @view(cb.values[axcb, :]) 
        # the constraint is (the a-var in c) = (the a-var in cb)
        for mxc in 2:Mc
            if c.sets[1, axc, mxc] == 0 continue end
            @. @view(constsMap.values[1, :]) += @view(constsMap.values[mxc, :])
        end

        # now the constraint is (the a-var in cb) = 0
        # it is a linear diophantine equation
        print("The constraint on cb is: ")
        print_mvars(stdout, constsMap.values[1,:])
        print(" = 0\n")

        if !avar_occupied(constsMap, 1)
            # the equation is 0=0, which is compatible, no new information gained
            continue
        end

        if avar_constant(constsMap, 1)
            # the equation is 0=c, incompatible, everything is wrong
            return 1, axcb
        end

        if !any(x -> abs(x) == 1, @view(constsMap.values[1, 2:end]))
            code, axerr = solve_dioph(c, cb, constsMap, axc)
            if code != 0
                return code, axerr
            end
            continue
        end

        # if there are any variables with coefficients equal to 1, it is pretty easy to solve
        # there is an extraneous variable wich is equal to the linear combination of thee other variables
        mxcb = findfirst(x -> abs(x) == 1, @view(constsMap.values[1, 2:end])) + 1
        coef = constsMap.values[1, mxcb]
        constsMap.values[1, mxcb] = 0
        if coef == 1
            @. @view(constsMap.values[1, :]) = - @view(constsMap.values[1, :])
        end
        # print("the result is: ")
        # print_mvars(stdout, begin x = zeros(Int32, mvars_size(cb)); x[mxcb] = 1; x end)
        # print(" = ")
        # print_mvars(stdout, @view(constsMap.values[1, :]))
        # print("\n")
        # print(cb)
        replace_var!(cb, mxcb, @view(constsMap.values[1, :]))
        replace_var!(constsMap, mxcb, @view(constsMap.values[1, :]))
        # print(cb)
    end

    return 0, 0
end

function should_innersubst(c::Constraint, cb::CombinedConstraints)
    Ac = avars_size(c)
    for axc in 1:Ac
        if avar_constant(cb, c.tuple[axc])
            return true 
        end
    end
    return false
end

function innersubst(c::Constraint, cb::CombinedConstraints)
    Ac = avars_size(c)
    Mc = mvars_size(c)
    S = sets_size(c)
    # this is an outsubst problem
    virtual_cb = CombinedConstraints(Ac, Mc)
    println(c.tuple, "  ", c.sets)
    virtual_tuple = filter((axc) -> avar_constant(cb, c.tuple[axc]), eachindex(c.tuple))
    println(virtual_tuple)
    virtual_c = Constraint(
        virtual_tuple,
        reshape(map(axc -> cb.values[c.tuple[axc], 1], virtual_tuple), (1, length(virtual_tuple), 1))
    )
    println(virtual_c)
    errax = 0
    errcode = 0

    newS = 0
    newM = 1
    newA = Ac - length(virtual_tuple)
    for sx in 1:S
        @. virtual_cb.values = @view(c.sets[sx, :, :])
        errcode, errax = outsubst(virtual_c, virtual_cb)
        if errcode != 0
            continue
        end
        newS += 1
        @. @view(c.sets[newS, :, :]) = virtual_cb.values
        Ms = renormalize_mvars!(@view(c.sets[newS, :, :]))
        newM = max(newM, Ms)
    end
    if newS == 0
        # we have a special case here, when there is a constraint that completely contradicts what we have
        return 100, c.tuple[errax] 
    end

    # we now shrink the constraint
    new_sets = Array{Int32, 3}(undef, newS, newA, newM)
    new_tuple = Vector{Int32}(undef, newA)
    axxv = 1
    for axc in eachindex(c.tuple)
        if axxv <= length(virtual_tuple) && axc == virtual_tuple[axxv]
            axxv += 1
        else
            new_tuple[axc-axxv+1] = axc
        end
    end
    for sx in 1:newS, ax in 1:newA
        @. @view(new_sets[sx, ax, :]) = @view(c.sets[sx, new_tuple[ax], 1:newM])
    end
    map!(axx -> c.tuple[axx], new_tuple, new_tuple)
    c.tuple = new_tuple
    c.sets = new_sets
    print(c)
    return 0, 0
end


function shape_inference_iteration(cs::Vector{Constraint}, cb::CombinedConstraints; should_print::Val{_print}=Val(false)) where {_print}
    
    things_did = 1
    while things_did != 0
        things_did = 0
        
        # outer substitution
        for ci in eachindex(cs)
            if sets_size(cs[ci]) == 1
                things_did += 1
                if _print
                    print("---------------------------\nDoing outside substitution for ")
                    print(cs[ci])
                    print("\n")
                end
                code, ax = outsubst(cs[ci], cb)
                if code != 0
                    error("Error code $(code), conflicting a-variable $(ax)")
                end
                cs[ci] = zero(Constraint)
                if _print
                    print(cb)
                    print(cs)
                    print("\n")
                end
            end
        end

        # inner substitution
        for ci in eachindex(cs)
            if should_innersubst(cs[ci], cb)
                if _print
                    print("---------------------------\nDoing inside substitution for ")
                    print(cs[ci])
                    print("\n")
                end
                things_did += 1
                code, ax = innersubst(cs[ci], cb)
                if code != 0
                    error("Error code $(code), conflicting a-variable $(ax)")
                end
                if _print
                    print(cb)
                    print(cs)
                    print("\n")
                end
            end
        end

        code, ax = check_valid(cb)
        if code != 0
            error("Error code $(code), conflicting a-variable $(ax)")
        end

    end
    renormalize_mvars!(cb)

end


# ----------------------
# PRINTING
# ----------------------

@inline print_delimited(io, print_element, delim, v::AbstractVector) = begin
    printed = false
    for i in eachindex(v)
        if printed print(io, delim) end
        print_element(io, v[i])
        printed = true
    end
end
@inline print_parensd(io, print_element, delim, v::AbstractVector) = begin
    if length(v) > 1 print(io, "(") end
    print_delimited(io, print_element, delim, v)
    if length(v) > 1 print(io, ")") end
end

print_mvars(io, r::AbstractVector{Int32}) = begin
    printed = false
    for i in eachindex(r)
        if r[i] == 0 continue end
        letter = (printed ? (r[i] < 0 ?  " - " : " + ") : (r[i] < 0 ? "-" : "")) * 
            (abs(r[i]) != 1 || i == 1 ? "$(abs(r[i]))" : "") *
            (i != 1 ? "nmlkopqrstuvwxyzbcdefghij"[mod(i-2,25)+1] : "") *
            (i > 24 ? "$(div(i-2,25)+1)" : "")
        print(io, letter)
        printed = true
    end
end

Base.show(io::IO, c::Constraint) = begin
    if length(c.tuple) == 0
        print(io, "EMPTY")
        return
    end
    A = avars_size(c)
    S = sets_size(c)
    print_parensd(io, (io, av) ->  print(io, "a$(av)"), ", ", c.tuple)
    print(io, " ∈ ")
    print_delimited(io, (io, set) -> begin
        print_parensd(io, print_mvars, ", ", [set[ax,:] for ax in 1:A])
    end, " ∪ ", [c.sets[sx,:,:] for sx in 1:S])
end

Base.show(io::IO, cb::CombinedConstraints) = begin
    print(io, "COMBINED: $(avar_count(cb)) entries\n")
    for ax in axes(cb.values, 1)
        if !avar_occupied(cb, ax) continue end
        print(io, "  a$(ax) = ")
        print_mvars(io, cb.values[ax,:])
        print(io, "\n")
    end
    print(io, "\n")
end

Base.show(io::IO, cs::Vector{Constraint}) = begin
    print(io, "OUTER: $(length(cs)) entries\n")
    print_delimited(io, (io, c) -> begin  
        print(io, "  ")
        print(io, c)
    end, "\n", cs)
    print("\n\n")
end

# --------------------
# FINAL FORM
# -------------------

macro make_constraint(tuple, sets...)
    if tuple.head != :tuple || length(sets) == 0
        !all(set -> set.head == :tuple, sets) || 
        !all(set -> length(set.args) == length(tuple.args), sets)
        error("The constraint has incorrect form. The correct form is: \n" *
            "(tuple of integers representing the indices in the array of shapes (a-vars)) = sets...\n" *
            "with each set being a tuple of same size containg linear expressions using constants and arbitrarily named variables (m-vars)")
    end
    
    mvars_dicts = map(set -> begin
        # get number of variables
        d = Dict{Symbol, Int32}()
        function traverse(expr)
            if expr isa Symbol
                if !(expr in keys(d)) && !(expr in (:+, :-, :*))
                    d[expr] = length(d)+2
                end
            elseif expr isa Expr
                for arg in expr.args
                    traverse(arg)
                end
            end
        end
        traverse(set)
        return d
    end, sets)
    S = length(sets)
    A = length(tuple.args)
    M = maximum(d -> length(d)+1, mvars_dicts)
    finalsets = zeros(Int32, (S, A, M))

    function show_err()
        error("The expressions inside the set must be a linear combination of constants and variables" * 
        ", so the only allowed operations are +, - and *. You also can't multiply variables together.")
    end

    function lc_eval(sx, expr) # it returns an vector of size M being the linear combination of those
        if expr isa Symbol
            toret = zeros(Int32, M)
            toret[mvars_dicts[sx][expr]] = 1
            return toret
        elseif expr isa Integer
            toret = zeros(Int32, M)
            toret[1] = expr
            return toret
        elseif expr isa Expr
            if expr.head != :call
                show_err()
            end
            if expr.args[1] == :+
                return lc_eval(sx, expr.args[2]) .+ lc_eval(sx, expr.args[3])
            elseif expr.args[1] == :-
                return lc_eval(sx, expr.args[2]) .+ lc_eval(sx, expr.args[3])
            elseif expr.args[1] == :*
                va = lc_eval(sx, expr.args[2])
                vb = lc_eval(sx, expr.args[3])
                varsa = !all(==(0), @view(va[2:end]))
                varsb = !all(==(0), @view(vb[2:end]))
                if varsa && varsb
                    show_err()
                elseif !varsa && !varsb
                    toret = zeros(Int32, M)
                    toret[1] = va[1] * vb[1]
                    return toret
                elseif !varsa && varsb
                    va, vb = vb, va
                end
                return va .* vb[1]
            else
                show_err()
            end
        else
            show_err()
        end
    end

    for sx in 1:S, ax in 1:A
        @view(finalsets[sx, ax, :]) .= lc_eval(sx, sets[sx].args[ax])
    end
    return quote
        $(esc(:Constraint))(
            $(esc(Expr(:ref, :(Int32), tuple.args...))),
            $finalsets
        )
    end
end

function push_constraints_broadcast(cs::AbstractVector{Constraint}, (outoffset, loffset)::NTuple{2,<:Integer}, ::Val{N}) where {N}
    for nx in 1:N
        push!(cs, @make_constraint((outoffset+nx, loffset+nx), (n, n)))
    end
end

function push_constraints_broadcast(cs::AbstractVector{Constraint}, (outoffset, loffset, roffset)::NTuple{3,<:Integer}, ::Val{N}) where {N}
    for nx in 1:N
        push!(cs, @make_constraint((outoffset+nx, loffset+nx, roffset+nx), (n, 1, n), (n, n, 1), (n, n, n)))
    end
end

function shape_inference(
    tree::TensorNode{T,N},
    operators::TensorOperatorEnum,
    cX::FlattenedTensorList{T,N},
) where {N,T}

    # now we have indices
    renumber_nodes!(tree)
    A = number_of_indices(tree)*N
    cb = CombinedConstraints(A, 5)
    cs = Constraint[]
    sizehint!(cs, A)

    function traverse(node)
        if node.degree == 0
            if !node.constant
                push!(cs, Constraint(
                    collect((node.index-1)*N .+ (1:N)),
                    reshape(collect(cX.positions[node.feature][3]), (1, N, 1))
                ))
                println("appending for ", node)
            end
        elseif node.degree == 1
            traverse(node.l)
            operators.unaops[node.op].push_constraints!(cs, ((node.index-1)*N, (node.l.index-1)*N), Val(N))
            println("appending for ", node)
        elseif node.degree == 2
            traverse(node.l)
            traverse(node.r)
            operators.binops[node.op].push_constraints!(cs, ((node.index-1)*N, (node.l.index-1)*N, (node.r.index-1)*N), Val(N))
            println("appending for ", node)
        end
    end
    push!(cs, Constraint(
        collect((tree.index-1)*N .+ (1:N)),
        reshape(collect(cX.positions[length(cX.positions)-1][3]), (1, N, 1))
    ))
    traverse(tree)

    println("--------- INITIAL SITUATION ------------")
    print(cb)
    print(cs)

    shape_inference_iteration(cs, cb; should_print=Val(true))

    # nodes = TensorNode{T,N}[]
    # flatten_tree!(nodes, tree)
    # if throw_errors
    #     _shape_inference(nodes, operators, Val(M), Val(C), featureSizes)
    #     return true
    # else
    #     return try
    #         _shape_inference(nodes, operators, Val(M), Val(C), featureSizes)
    #         return true
    #     catch e
    #         return false
    #     end
    # end
end


end