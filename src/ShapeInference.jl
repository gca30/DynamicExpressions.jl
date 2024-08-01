
module ShapeInferenceModule

using ...NodeModule: TensorNode
using ..OperatorEnumModule: TensorOperatorEnum
using ..FlattenedTensorListModule: FlattenedTensorList
using ...NodeUtilsModule: recalculate_node_indices!, number_of_indices

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
@inline mvar_occupied(c::Constraint, mx) = any(sx -> mvar_occupied(@view(c.sets[sx, :, :]), mx), axes(c.sets, 1))
@inline avar_occupied(cb::CombinedConstraints, ax) = any(!=(0), @view(cb.values[ax,:]))
@inline avar_constant(cb::CombinedConstraints, ax) = all(==(0), @view(cb.values[ax,2:end]))
@inline avar_nzconstant(cb::CombinedConstraints, ax) = cb.values[ax,1] != 0 && avar_constant(cb, ax)
@inline avar_constant(ar::AbstractArray) = all(==(0), @view(ar[2:end]))
@inline avar_constant(c::Constraint, sx, ax) = all(==(0), @view(c.sets[sx,ax,2:end]))
@inline avar_count(cb::CombinedConstraints) = count(ax -> avar_occupied(cb, ax), axes(cb.values, 1))
@inline mvar_count(cb::CombinedConstraints) = count(mx -> mvar_occupied(cb, mx), 2:size(cb.values, 2))+1
@inline mvar_count(cbv::AbstractMatrix) = count(mx -> mvar_occupied(cbv, mx), 2:size(cbv, 2))+1
@inline Base.isempty(c::Constraint) = length(c.sets) == 0

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

function renormalize_mvars!(c::Constraint)
    tmx = 0
    for sx in axes(c.sets, 1)
        nmx = renormalize_mvars!(@view(c.sets[sx,:,:]))
        tmx = max(tmx, nmx)
    end
    c.sets = copy(@view(c.sets[:,:,1:tmx]))
    return tmx
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

        @. @view(constsMap.values[1, :]) = -@view(cb.values[axcb, :])
        # the constraint is (the a-var in c) = (the a-var in cb)
        constsMap.values[1, 1] += c.sets[1, axc, 1]
        for mxc in 2:Mc
            if c.sets[1, axc, mxc] == 0 continue end
            @. @view(constsMap.values[1, :]) += @view(constsMap.values[mxc, :])
        end

        # now the constraint is (the a-var in cb) = 0
        # it is a linear diophantine equation
        # print("The constraint on cb is: ")
        # print_mvars(stdout, constsMap.values[1,:])
        # print(" = 0\n")

        if avar_constant(constsMap, 1)
            if constsMap.values[1, 1] == 0
                continue
            else
                return 1, axcb
            end
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
        # println(cb)
        replace_var!(cb, mxcb, @view(constsMap.values[1, :]))
        replace_var!(constsMap, mxcb, @view(constsMap.values[1, :]))
        # println(cb)
    end

    return 0, 0
end

function should_innersubst(c::Constraint, cb::CombinedConstraints)
    Ac = avars_size(c)
    for axc in 1:Ac
        if avar_nzconstant(cb, c.tuple[axc])
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
    virtual_tuple = filter((axc) -> avar_nzconstant(cb, c.tuple[axc]), eachindex(c.tuple))
    virtual_c = Constraint(
        virtual_tuple,
        reshape(map(axc -> cb.values[c.tuple[axc], 1], virtual_tuple), (1, length(virtual_tuple), 1))
    )
    # println(virtual_c)
    errax = 0
    errcode = 0

    new_sets_u = Array{Int32, 3}(undef, S, Ac, Mc)
    newS = 0
    newM = 1
    newA = Ac - length(virtual_tuple)
    for sx in 1:S
        @. virtual_cb.values = @view(c.sets[sx, :, :])
        # println("WE HAVE: ", virtual_cb)
        # println("WITH ", virtual_c)
        errcode, errax = outsubst(virtual_c, virtual_cb)
        if errcode != 0
            continue
        end
        newS += 1
        @. @view(new_sets_u[newS, :, :]) = virtual_cb.values
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
        @. @view(new_sets[sx, ax, :]) = @view(new_sets_u[sx, new_tuple[ax], 1:newM])
    end
    map!(axx -> c.tuple[axx], new_tuple, new_tuple)
    c.tuple = new_tuple
    c.sets = new_sets
    # println("NEW VALUE: ", c)
    return 0, 0
end

# 0 if they are not subsets of each other
# 1 if s1 is a subset of s2
# 2 if s2 is a subset of s1
# 3 if they are equivalent
function subset_comparison(c::Constraint, s1::Integer, s2::Integer)
    # s1 is a subset of s2 if we can find variables such that the equation is compatible
    # this can be done using outsubst (we say we have a set that is the combined constraints and a set that is )
    Mb = mvars_size(c)
    Ab = avars_size(c)
    virtual_cb = CombinedConstraints(Ab, Mb)
    virtual_cb.values .= @view(c.sets[s1, :, :])
    virtual_c = Constraint(collect(1:Ab), c.sets[s2:s2,:,:])
    initial_mvars_1 = mvar_count(@view(c.sets[s1,:,:]))
    initial_mvars_2 = mvar_count(@view(c.sets[s2,:,:]))
    ercode, _ = outsubst(virtual_c, virtual_cb)
    if ercode != 0 # not compatible
        return 0
    elseif initial_mvars_1 == initial_mvars_2 
        # the dimensionality is the same, and if they are compatible, it means they are equivalent
        return 3
    elseif initial_mvars_1 < initial_mvars_2
        return 1
    else
        return 2
    end
    
end

function simplify(cs::AbstractVector{Constraint}, ci::Integer)
    # simplifies constraints that have redundant sets
    Sb = sets_size(cs[ci])
    if Sb < 2 return 0, 0 end
    Mb = mvars_size(cs[ci])
    Ab = avars_size(cs[ci])
    
    # for every set, we check if there is a set that is a subset of the original
    # if it is, we remove it
    save_map = zeros(Int32, Sb)
    save_map[1] = 1
    save_count = 1
    for sx in 2:Sb
        save_flag = true
        for j in 1:(sx-1)
            sc = subset_comparison(cs[ci], sx, j)
            if sc == 0 # incompatible
                # save the set if there are only incompatibilities
                continue
            elseif sc == 1 || sc == 3 # sx subset of j or equivalent set
                # don't save the set
                # do nothing with it
                save_flag = false
                break  
            elseif sc == 2 # j subset of sc
                # replace the set with sc
                # don't save as a new set
                save_map[sx] = save_map[j]
                save_map[j] = 0
                save_flag = false
                break
            end
            # if they are equal we must onyl remove 1
            #if j > sx && @view(cs[ci].sets[sx, :, :]) == @view(cs[ci].sets[j, :, :]) continue end 
        end

        if save_flag
            save_map[sx] = save_count+1
            save_count += 1
        end
    end

    new_sets = Array{Int32, 3}(undef, save_count, Ab, Mb)
    for savex in eachindex(save_map)
        if save_map[savex] == 0 continue end
        @view(new_sets[save_count,:,:]) .=  @view(cs[ci].sets[save_map[savex],:,:])
        save_count -= 1
    end
    cs[ci].sets = new_sets
    renormalize_mvars!(cs[ci])
    for sx in axes(cs[ci].sets, 1), ax in axes(cs[ci].sets, 2)
        if avar_constant(cs[ci], sx, ax) && cs[ci].sets[sx,ax,1] < 0
            return 33, ci
        end
    end

    return 0, 0
end

function split_by_variables(cs::AbstractVector{Constraint}, ci::Integer)
    Sb = sets_size(cs[ci])
    if Sb < 2 return 0, 0 end
    Mb = mvars_size(cs[ci])
    Ab = avars_size(cs[ci])
    avars_map = zeros(Int32, Ab)
    mvars_map = zeros(Int32, Mb-1) # what constraint does the mvar fall in (0 for not yet determined)
    ccount = 0
    count_minus1 = 0
    for sx in 1:Sb, ax in 1:Ab
        if avars_map[ax] != -1 && cs[ci].sets[sx, ax, 1] != cs[ci].sets[1, ax, 1]
            avars_map[ax] = -1 # means that we should include it in every set, except the constants one
            count_minus1 += 1
        end
    end

    for sx in 1:Sb
        mvars_map .= 0
        for ax in 1:Ab, mx in 2:Mb
            if cs[ci].sets[sx, ax, mx] == 0
                continue
            end
            mm = mvars_map[mx-1]
            am = avars_map[ax]
            if mm == 0 && am != 0
                mvars_map[mx-1] = am
            elseif mm != 0 && am == 0
                avars_map[ax] = mm
            elseif mm == 0 && am == 0
                avars_map[ax] = ccount + 1
                mvars_map[mx-1] = ccount + 1
                ccount += 1
            elseif mm != am && am != -1
                toreplace = max(mm, am)
                replacedwith = min(mm, am)
                replace!(x -> x == toreplace ? replacedwith : (x > toreplace ? x-1 : x), mvars_map)
                replace!(x -> x == toreplace ? replacedwith : (x > toreplace ? x-1 : x), avars_map)
                ccount -= 1
            elseif mm != am && am == -1
                replace!(x -> x == mm ? -1 : (x > mm ? x-1 : x), mvars_map)
                ccount -= 1 
            end
        end
    end

    # when avars_map contains -1, means that we have constants that differ depending on the case
    # when avars_map contains 0, it means that it is a constant that is the same across the field
    # when avars_map contains >0, it means that we can split in evenly
    count_0 = count(==(0), avars_map)
    if count_0 != 0
        new_tuple = filter(x -> avars_map[x] == 0, eachindex(avars_map))
        new_sets = Array{Int32, 3}(undef, 1, length(new_tuple), 1)
        for nax in eachindex(new_tuple)
            new_sets[1, nax, 1] = cs[ci].sets[1, new_tuple[nax], 1]
        end
        replace!(x -> cs[ci].tuple[x], new_tuple)
        push!(cs, Constraint(
            new_tuple, new_sets
        ))
    end
    for i in 1:ccount
        new_tuple = filter(x -> avars_map[x] == i || avars_map[x] == -1, eachindex(avars_map))
        new_sets = Array{Int32, 3}(undef, Sb, length(new_tuple), Mb)
        for nax in eachindex(new_tuple)
            @view(new_sets[:, nax, :]) .= @view(cs[ci].sets[:, new_tuple[nax], :])
        end
        replace!(x -> cs[ci].tuple[x], new_tuple)
        push!(cs, Constraint(
            new_tuple, new_sets
        ))
        erc, erv = simplify(cs, length(cs))
        if erc != 0
            return erc, erv
        end
    end
    if ccount == 0
        new_tuple = filter(x -> avars_map[x] == -1, eachindex(avars_map))
        new_sets = Array{Int32, 3}(undef, Sb, length(new_tuple), Mb)
        for nax in eachindex(new_tuple)
            @view(new_sets[:, nax, :]) .= @view(cs[ci].sets[:, new_tuple[nax], :])
        end
        replace!(x -> cs[ci].tuple[x], new_tuple)
        push!(cs, Constraint(
            new_tuple, new_sets
        ))
        erc, erv = simplify(cs, length(cs))
        if erc != 0
            return erc, erv
        end
    end
    cs[ci] = zero(Constraint)

    return 0,0
end

function split_and_simplify(cs::AbstractVector{Constraint}, ci::Integer)
    return split_by_variables(cs, ci)
end

function shape_inference_iteration(cs::Vector{Constraint}, cb::CombinedConstraints)
    
    things_did = 1
    while things_did != 0
        things_did = 0
        
        # outer substitution
        for ci in eachindex(cs)
            if sets_size(cs[ci]) == 1
                things_did += 1
                # print("---------------------------\nDoing outside substitution for ")
                # println(cs[ci])
                code, ax = outsubst(cs[ci], cb)
                if code != 0
                    error("Error code $(code), conflicting a-variable $(ax)")
                end
                cs[ci] = zero(Constraint)
                # println(cb)
                # println(cs)
            end
        end

        # inner substitution
        for ci in eachindex(cs)
            if should_innersubst(cs[ci], cb)
                # print("---------------------------\nDoing inside substitution for ")
                # println(cs[ci])
                things_did += 1
                code, ax = innersubst(cs[ci], cb)
                if code != 0
                    error("Error code $(code), conflicting a-variable $(ax)")
                end
                code, ax = split_and_simplify(cs, ci)
                if code != 0
                    error("Error code $(code), conflicting a-variable $(ax)")
                end
                # println(cb)
                # println(cs)
            end
        end

        if things_did > 0
            filter!(c -> !isempty(c), cs)
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
    #print(io, "\n")
end

Base.show(io::IO, cs::Vector{Constraint}) = begin
    print(io, "OUTER: $(length(cs)) entries\n")
    print_delimited(io, (io, c) -> begin  
        print(io, "  ")
        print(io, c)
    end, "\n", cs)
    # print("\n\n")
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
                toret = zeros(Int32, M)
                for i in 2:length(expr.args)
                    toret .+= lc_eval(sx, expr.args[i])
                end
                return toret
            elseif expr.args[1] == :-
                if length(expr.args) == 3
                    return lc_eval(sx, expr.args[2]) .- lc_eval(sx, expr.args[3])
                elseif length(expr.args) == 2
                    return .- lc_eval(sx, expr.args[2])
                else
                    show_err()
                end
            elseif expr.args[1] == :*
                vv = map(ex -> lc_eval(sx, ex), expr.args[2:end])
                count_vv = count(ve -> !avar_constant(ve), vv)
                if count_vv == 0
                    toret = zeros(Int32, M)
                    toret[1] = prod(ve -> ve[1], vv)
                    return toret
                elseif count_vv == 1
                    varx = findfirst(ve -> !avar_constant(ve), vv)
                    for i in eachindex(vv)
                        if i == varx continue end
                        vv[varx] .*= vv[i][1]
                    end
                    return vv[varx]
                else
                    show_err()
                end
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

function default_constraint_indeterminancy_resolve(c::Constraint, node::TensorNode{T,N}, dim::Integer, operators::TensorOperatorEnum) where {T,N}
    S = sets_size(c)
    s = rand(UInt32)%S+1
    return Constraint(c.tuple, c.sets[s:s,:,:])
end

function shape_inference(
    tree::TensorNode{T,N},
    operators::TensorOperatorEnum,
    cX_shapes::AbstractVector{NTuple{N, <:Integer}},
    desired_shape::NTuple{N, <:Integer};
    kws...
) where {N,T}
    push!(cX_shapes, desired_shape)
    shape_inference(tree, operators, cX_shapes; kws...)
    pop!(cX_shapes)
end

function shape_inference(
    tree::TensorNode{T,N},
    operators::TensorOperatorEnum,
    cX::Union{FlattenedTensorList{T,N}, AbstractVector{NTuple{N, <:Integer}}};
    indeterminancy_resolve::F1 = default_constraint_indeterminancy_resolve,
) where {N,T,F1}

    # now we have indices
    recalculate_node_indices!(tree)
    A = number_of_indices(tree)*N
    cb = CombinedConstraints(A, 5)
    cs = Constraint[]
    sizehint!(cs, A)

    function traverse(node)
        if node.degree == 0
            if !node.constant
                push!(cs, Constraint(
                    collect((node.index-1)*N .+ (1:N)),
                    reshape(collect(if cX isa FlattenedTensorList 
                        cX.positions[node.feature].shape
                    else
                        cX[node.feature]
                    end), (1, N, 1))  
                ))
                # println("appending for ", node, " : ", cs[length(cs)])
            end
        elseif node.degree == 1
            traverse(node.l)
            # i = length(cs)+1
            operators.unaops[node.op].push_constraints!(cs, ((node.index-1)*N, (node.l.index-1)*N), Val(N))
            # print("appending for ", node, " : ")
            # for ix in i:length(cs)
            #     print(cs[ix], "   ")
            # end
            # print("\n")
        elseif node.degree == 2
            traverse(node.l)
            traverse(node.r)
            # i = length(cs)+1
            operators.binops[node.op].push_constraints!(cs, ((node.index-1)*N, (node.l.index-1)*N, (node.r.index-1)*N), Val(N))
            # print("appending for ", node, " : ")
            # for ix in i:length(cs)
            #     print(cs[ix], "   ")
            # end
            # print("\n")
        end
    end
    function get_node_by_index(node, index)
        if node.degree == 0
            if node.index == index
                return node
            end
        elseif node.degree == 1
            if index == node.index
                return node
            elseif index <= node.l.index
                return get_node_by_index(node.l, index)
            end
        elseif node.degree == 2
            if index <= node.l.index
                return get_node_by_index(node.l, index)
            elseif index <= node.r.index
                return get_node_by_index(node.r, index)
            elseif index == node.index
                return node
            end
        end
    end
    push!(cs, Constraint(
        collect((tree.index-1)*N .+ (1:N)),
        reshape(collect(
            if cX isa FlattenedTensorList 
                cX.positions[length(cX.positions)-1].shape
            else
                cX[length(cX)]
            end
        ), (1, N, 1))
    ))
    traverse(tree)

    # println("--------- INITIAL SITUATION ------------")
    # println(cb)
    # println(cs)

    while true
        shape_inference_iteration(cs, cb)
        # println("------- PARTIALLY FINAL SITUATION ------------")
        # println(cb)
        # println(cs)
        if length(cs) != 0
            # node = get_node_by_index()
            node = get_node_by_index(tree, div(cs[1].tuple[1], N))
            S = sets_size(cs[1])
            s = rand(UInt32)%S+1
            cs[1] = Constraint(cs[1].tuple, cs[1].sets[s:s,:,:])
            #cs[1] = indeterminancy_resolve(cs[1], node, mod(cs[1].tuple[1], N), operators)
        elseif mvars_size(cb) != 1
            z = zeros(Int32, mvars_size(cb))
            z[1] = rand(UInt32)%5+1
                # TODO: create some sort of generator, given the already used shape, the operator, the nodes, etc
                # maybe each operator should have a default generator that is random?
                # this also means that
            replace_var!(cb, 2, z)
        else
            break
        end
    end

    # println("------- FULLY FINAL SITUATION ------------")
    # println(cb)
    # println(cs)

    function final_traverse(node)
        if node.degree == 1
            final_traverse(node.l)
        elseif node.degree == 2
            final_traverse(node.l)
            final_traverse(node.r)
        end
        node.shape = ntuple(i -> cb.values[(node.index-1)*N+i], Val(N))
    end
    final_traverse(tree)

end


end