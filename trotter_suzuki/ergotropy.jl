#include("gear_init.jl")
# import JLD: save, @save, load, @load

# First, let's write a function to compute ρ₂ from Cm state, by tracing out Gear 1.
function rho2fromCm(Cm::Array{T,2}) where T<:Number
    result = transpose(ctranspose(Cm)*Cm)
    @assert abs(trace(result)-1)<1e-8
    return result
end

function get_index(m, m_min)
    return m - m_min + 1
end

function get_m(ind, m_min)
    return ind + m_min - 1
end

function check_ascending_order(arr::Array{T,1}) where T<:Real
    N_arr = length(arr)
    flag = true
    for i=1:(N_arr-1)
        if arr[i+1]-arr[i] < -1e-8
            flag = false
            break
        end
    end
    return flag
end

"""
            k2ij

    Take a index k for a kron tensor product vector, and return the (i,j) index
    pair of the tensor product.
"""
function k2ij(k::Int,N2::Int)
    i = div(k-1,N2) + 1
    j = mod(k-1,N2) + 1
    return (i,j)
end

"""
            v2Cm

    Convert a "kron" tensor-product state to a Cm::StateMatrix format.
"""
function v2Cm(v::Array,Npoints::Int)
    Cm = zeros(Complex128,Npoints,Npoints)
    for k=1:Npoints^2
        ind1,ind2 = k2ij(k,Npoints)
        # m1 = ind1 - 1 + m_min
        # m2 = ind2 - 1 + m_min
        Cm[ind1,ind2] = v[k]
    end
    return Cm
end #function
# Now, a function to compute the ergotropy given the state
# For simplicity, let's assume that m_min == -m_max.
"""
                ergotropy

    Compute the ergotropy of a density matrix of Gear 2.
    Assume:
              m_min == -m_max
"""
function ergotropy(rho2::Array{T,2}, m_min::Int, m_max::Int) where T<:Number
    L_gear_square = spdiagm(Float64.(mpoints_gear.^2))
    NDim = size(rho2)[1]
    @assert NDim == m_max-m_min+1
    @assert m_min == -m_max
    rho2 = Hermitian(rho2)
    D_eigs, = eig(rho2)
    @assert check_ascending_order(D_eigs) == true
    @assert abs(sum(D_eigs)-1)<1e-8
    D_eigs = D_eigs[end:-1:1]
    arr_passive = zeros(Complex128,NDim)
    # assemble the ground state
    ind = get_index(0,m_min)
    arr_passive[ind] = D_eigs[1]
    # other states
    for m = 1:m_max
        ind = get_index(-m,m_min)
        ind_eig = 2*m
        arr_passive[ind] = D_eigs[ind_eig]

        ind = get_index(m, m_min)
        ind_eig = m*2 + 1
        arr_passive[ind] = D_eigs[ind_eig]
        # In principle, it shouldn't matter whether we put smaller value to -m or m? Check with guys
    end
    rho_passive = diagm(arr_passive)
    rho_diff = rho_passive - rho2
    result = trace(L_gear_square*rho_diff)/(2*I2)
    @assert imag(result)<1e-6
    return -real(result)
end
