
# Second try at making a shape inference algorithm
# First iteration supported binary operator broadcasting, unary operator broadcasting, 
#   permutedims operation (transpose), cross products, etc..
# Second iteration should support concatenation, convolution, 
#  ^ These operation require that the dimensions are added or subtracted between eachother
#    The representation will be the same: list of constraints of the form 
#       (tuple of dimension) = union of sets of possible tuples formed by constants and shape variables
#       this time, there can also be arithmetic operations in the tuple or the parameters, leading to each constraint
#       element being a linear combination of variables and constants


using StaticArrays

# M - tuple size / number of a-variables
# C - number of sets
# V - number of variables in set + 1 (as the first row is constants)
# we lose type sfaety because it's too much pain to deal with
mutable struct Constraint{M, C, V}
    tuple :: MVector{M, Int32}
    sets :: MVector{C, MMatrix{M, V, Int32}}
end

@inline Base.zero(::Type{Constraint}) = Constraint{0,0,0,0}(zero(MVector{0, Int32}), zero(MVector{0, MMatrix{0,0,Int32}}))
@inline print_delimited(io, delim, v::AbstractVector{T}) where {T} = begin
    for i in eachindex(v)
        print(io, v[i])
        if i != length(v) print(io, delim) end
    end
end
print_vars(io, r::AbstractVector{Int32}) = begin
    printed = false
    for i in eachindex(r)
        if r[i] == 0 continue end
        letter = ""
        if i != 1
            letter *= "nmlkopqrstuvwxyzbcdefghij"[mod(i-2,25)+1]
            if i > 24
                letter *= "$(div(i-2,25)+1)"
            end
        end
        prefix = if printed && r[i] < 0
            " - "
        elseif printed && r[i] > 0
            " + "
        elseif !printed && r[i] < 0
            "-"
        elseif !printed && r[i] > 0
            ""
        end
        if abs(r[i]) != 1 || i == 1
            prefix *= "$(abs(r[i]))"
        end
        print(io, "$(prefix)$(letter)")
        printed = true
    end
end
Base.show(io::IO, c::Constraint{M,C,V}) where {M,C,V} = begin
    if M == 0
        print(io, "EMPTY")
        return
    end
    if M > 1 print(io, "(") end
    print_delimited(io, ", ", map(ax -> "a$(ax)", c.tuple))
    if M > 1 print(io, ")") end
    print(io, " ∈ ")
    print_delimited(io, " ∪ ", map(set -> begin
        buf = IOBuffer()
        if M > 1 print(buf, "(") end
        for i in 1:M
            if i != 1 print(buf, ", ") end
            print_vars(buf, set[i,:])
        end
        if M > 1 print(buf, ")") end
        String(take!(buf))
    end, c.sets))
end

# the outside is stored as:
mutable struct CombinedConstraints
    values :: Matrix{Int32}
    CombinedConstraints(A) = new(zeros(Int32, min(A, 10), A))
end

Base.show(io::IO, cb::CombinedConstraints) = begin
    print(io, "COMBINED: $(count(i -> any(!=(0), cb.values[:,i]), axes(cb.values, 2))) entries\n")
    for i in axes(cb.values, 2)
        if all(==(0), cb.values[:,i]) continue end
        print(io, "  a$(i) = ")
        print_vars(io, cb.values[:,i])
        print(io, "\n")
    end
    print(io, "\n")
end

@inline varvecsize(cb::CombinedConstraints) = size(cb.values, 1)
@inline expand(mat::Matrix) = cat(mat, zeros(eltype(mat), size(mat)...); dims=1)
@inline expand!(cb::CombinedConstraints) = begin
    cb.values = expand(cb.values)
end

first_unused_var!(cb::CombinedConstraints) = begin
    for i in axes(cb.values, 1)
        if i == 1 continue end
        if all(==(0), @view(cb.values[i,:])) return i end
    end
    v = length(cb.occupied) + 1
    expand!(cb)
    return v
end
first_unused_var!(cb::CombinedConstraints, constsMap::Matrix{Int32}) = begin
    for i in axes(cb.values, 1)
        if i == 1 continue end
        if all(==(0), @view(cb.values[i,:])) && all(==(0), @view(constsMap[i,:])) return i end
    end
    v = length(cb.occupied) + 1
    expand!(cb)
    return v
end
@inline ax_occupied(cb::CombinedConstraints, ax) = any(!=(0), @view(cb.values[ax,:]))
@inline ax_constant(cb::CombinedConstraints, ax) = cb.values[ax,1] != 0 && all(==(0), @view(cb.values[ax,2:end])) 

@inline should_outsubst(::Constraint{M,C,V}) where {M,C,V} = C == 1

function replace_var!(mat::Matrix{Int32}, ix, val::Vector{Int32})
    for i in size(mat, 2)
        if mat[ix, i] == 0 continue end
        coef = mat[ix, i]
        mat[ix, i] = 0
        @. @view(mat[:,i]) += val * coef
    end
end

function check_valid(cb::CombinedConstraints)
    for ax in axes(cb.values, 2)
        if any(!=(0), @view(cb.values[2:end,ax])) continue end
        if cb.values[1,ax] < 0
            return 3, ax
        end
    end
    return 0, 0
end

#= ::Tuple{Int32,Int32} -> if an error occurs, it returns the upsetting "error code", a_x =#
function outsubst(c::Constraint{M,C,V}, cb::CombinedConstraints) where {M,C,V}
    currentVars = varvecsize(cb)
    # create the map for consts
    constsMap = zeros(Int32, currentVars, V)
    constsMap[1,1] = 1
    for i in 1:M
        ax = c.tuple[i]
        value = c.sets[1][i,:]
        
        if ax_occupied(cb, ax)
            
            # adding the constants to the combined constraints
            for j in 2:V
                if value[j] == 0 continue end
                if any(!=(0), constsMap[:,j])
                    newvar = first_unused_var!(cb, constsMap)
                    if newvar > currentVars
                        currentVars = varvecsize(cb)
                        constsMap = expand(constsMap)
                    end
                    constsMap[newvar, j] = 1
                end
            end

            result = zeros(Int32, currentVars)
            result[1] = value[1]
            for j in 2:V
                if value[j] == 0 continue end
                @. result += @view(constsMap[:,j]) * value[j]
            end
            @. result -= @view(cb.values[ax,:])
            
            # first, we check if it is a constant
            if all(==(0), @view(result[2:end]))
                if result[1] == 0
                    continue
                else
                    return 1, ax
                end

            # then we check that there is a constant here with 1 or -1 term
            # this is pretty common and is very easy to solve
            elseif any(x -> x == 1 || x == -1, @view(result[2:end]))
                vartoreplace = findfirst(x -> x == 1 || x == -1, @view(result[2:end]))+1
                if result[vartoreplace] == 1
                    result .*= -1
                end
                result[vartoreplace] = 0
                replace_var!(constsMap, vartoreplace, result)
                replace_var!(cb.values, vartoreplace, result)

                
            # now we must solve the diopantine equation
            # result \cdot [1 n_1 n_2 ... n_k]^T
            else

                cnt = count(!=(0), @view(result[2:end]))
                temp = Vector{MVector{4, Int32}}(undef, cnt)
                #             bc, u, v, index
                first = true
                for i in reverse(eachindex(result))
                    if result[i] == 0 || i == 1 continue end
                    if first
                        first = false
                        temp[cnt][1] = result[i]
                        temp[cnt][2] = 1
                        temp[cnt][3] = 0
                        temp[cnt][4] = i
                        cnt -= 1
                    else
                        g, u, v = gcdx(result[i], temp[cnt+1][1])
                        temp[cnt][1] = g
                        temp[cnt][2] = u
                        temp[cnt][3] = v
                        temp[cnt][4] = i
                        cnt -= 1
                    end
                end
                if mod(result[1], temp[1][4]) != 0
                    return 2, ax # no solutions error
                end

                K = first_unused_var!(cb, constsMap)
                if K > currentVars
                    currentVars = varvecsize(cb)
                    constsMap = expand(constsMap)
                end
                nc = zeros(Int32, currentVars)
                ncc = zeros(Int32, currentVars)
                nc[1] = -div(result[1], temp[1][4])
                for i in eachindex(temp)
                    if i == length(temp)
                       continue 
                    end 
                    @. ncc = nc * temp[i][2]
                    ncc[K] += div(temp[i+1][1], temp[i][1])
                    replace_var!(cb.values, temp[i][4], ncc)
                    replace_var!(constsMap, temp[i][4], ncc)
                    @. nc = nc * temp[i][3]
                    nc[K] -= div(temp[i][1], temp[i][1])
                    K = temp[i][4]
                    if i == length(temp)-1
                        replace_var!(cb.values, temp[i+1][4], nc)
                        replace_var!(constsMap, temp[i+1][4], nc)
                    end
                end

            end

        else
            cb.values[ax,1] = value[1]
            for j in 2:V
                if value[j] == 0 continue end                
                if any(!=(0), constsMap[:,j])
                    # get a new variable, extend if necesary
                    newvar = first_unused_var!(cb, constsMap)
                    if newvar > currentVars
                        currentVars = varvecsize(cb)
                        constsMap = expand(constsMap)
                    end
                    constsMap[newvar, j] = 1
                end

                @. @view(cb.values[ax,:]) += @view(constsMap[:,j]) * value[j]
            end
        end
    end
    return 0, 0
end

function innersubst(c::Constraint{M,C,V}, cb::CombinedConstraints) where {M,C,V}
    constsMap = MMatrix{V, V, Int32}
    changed = false
    for axi in c.tuple
        if !ax_constant(cb, c.tuple[axi]) continue end
        changed = true
        # 3n + 2m + 2 = 7 => 3n + 2m = 5
        #   => { n = 1 + 2k
        #      { m = 1 - 2k
        for seti in eachindex(c.sets)
            value = c.sets[seti][ax,2:end]
            cons = cb.values[ax,1] - value[1]

            # eq: value = ax[1][1]
        end
    end
    return c
end

function shape_inference_iteration(cs::Vector{Constraint}, cb::CombinedConstraints)
    
    # outer substitution
    for ci in eachindex(cs)
        if should_outsubst(cs[ci])
            code, ax = outsubst(cs[ci], cb)
            if code != 0
                error("Error code $(code), conflicting a-variable $(ax)")
            end
            cs[ci] = zero(Constraint)
        end
    end

    # inner substitution
    for ci in eachindex(cs)
        
    end

    code, ax = check_valid(cb)
    if code != 0
        error("Error code $(code), conflicting a-variable $(ax)")
    end
    # filter(!isempty, cs)
end

# EXAMPLE:

cb = CombinedConstraints(44)
cs = Constraint[]
# a1 = a2 is equivalent to:
push!(cs, Constraint{2,1,2}(
    MVector{2,Int32}([1, 2]),
    (MMatrix{2,2,Int32}([0 1; 0 1]),)
))
# a3 = 5 is equivalent to:
push!(cs, Constraint{1,1,1}(
    MVector{1,Int32}([3]),
    (MMatrix{1,1,Int32}([5;;]),)
))
# (a4,a5,a6) = {(n,1,n)} u {(n,n,n)} u {(n,n,1)} is:
push!(cs, Constraint{3,3,2}(
    MVector{3,Int32}([4, 5, 6]),
    (MMatrix{3,2}([0 1; 1 0; 0 1]), MMatrix{3,2}([0 1; 0 1; 0 1]), MMatrix{3,2}([0 1; 0 1; 1 0]))
))
# a9 = a7+a8 is:
push!(cs, Constraint{3,1,3}(
    MVector{3,Int32}([7, 8, 9]),
    (MMatrix{3, 3}([0 1 0; 0 1 0; 0 1 1]),)
))

print_cs(cs) = print_cs(stdout, cs)
print_cs(io, cs) = begin
    print(io, "OUTER: $(length(cs)) entries\n")
    print_delimited(io, "\n", map(c -> begin 
        buf = IOBuffer() 
        print(buf, "  ")
        print(buf, c)
        String(take!(buf)) 
    end, cs))
    print("\n\n")
end

cb.values[3, 3] = 3
cb.values[4, 5] = 6
cb.values[3, 5] = 3
cb.values[1, 5] = 2
cb.values[1, 1] = 9
cb.values[1, 2] = 9
cb.values[1, 2] = 1
print(cb)
print_cs(cs)