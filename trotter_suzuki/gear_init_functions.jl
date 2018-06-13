################################################################################

#  useful functions for computing time evolutions and statistics

# These functions may rely on the operators, arrays and numbers defined above.

################################################################################

function prepare_gaussian_m(m_list::Union{Array,Range}, m_mean::Real, sigma::Real)
    Cm = exp.(-(m_list.-m_mean).^2/(4*sigma^2))
    Cm = Cm/norm(Cm)
    return Array{Complex128}(Cm)
end

function mean_operator(Cm::Array{T,2}, ope::Array{T,2}) where T<:Number
    C2 = abs2.(circshift(Cm,[m_min,m_min]))
    return sum(C2.*ope)
end


function statesampling!(C_array::StateMatrix, statesample_list::Array{StateMatrix,1}, ind_sample::Int, m_min::Int)
    Cm12 = circshift(C_array,[-m_min,-m_min])
    statesample_list[ind_sample] = Cm12
end

function recordmeanvariance!(C_array::StateMatrix, mean_L1::Array{Float64,1}, mean_L2::Array{Float64,1}, variance_L1::Array{Float64,1}, variance_L2::Array{Float64,1}, ind::Int)
    C2 = similar(C_array)
    C_temp = similar(C_array)
    broadcast!(abs2,C2,C_array)
    broadcast!(*,C_temp,L1_ope,C2)
    @inbounds mean_L1[ind] = sum(C_temp)
    broadcast!(*,C_temp,L1square_ope,C2)
    @inbounds variance_L1[ind] = sum(C_temp) - mean_L1[ind]^2
    broadcast!(*,C_temp,L2_ope,C2)
    @inbounds mean_L2[ind] = sum(C_temp)
    broadcast!(*,C_temp,L2square_ope,C2)
    @inbounds variance_L2[ind] = sum(C_temp) - mean_L2[ind]^2
end

function infinitesimal_trotter_suzuki_solver(Cm0::StateMatrix, m_min::Int, m_max::Int, dt::Real)
    UKdt_half = exp.(-im*dt/2.0*K_ope)
    UVdt = exp.(-im*dt*V_ope)
    C_array = circshift(Cm0,[m_min,m_min])
    # do the time evolution
    C_array_temp = similar(C_array)
    broadcast!(*,C_array_temp,UKdt_half,C_array)
    T_xp! * C_array_temp
    broadcast!(*, C_array, UVdt, C_array_temp)
    T_px! * C_array
    C_array = UKdt_half .* C_array
    return circshift(C_array,[-m_min,-m_min])
end #function



function trotter_suzuki_solver(Cm0::StateMatrix, m_min::Int, m_max::Int,
                                t0::Real, tf::Real, dt::Real, nsample::Int)

    # prepare infinitesimal evolution operators
    UKdt_half = exp.(-im*dt/2.0*K_ope)
    UVdt = exp.(-im*dt*V_ope)
    # memory pre-allocation
    t_list = t0:dt:tf; Nt_list = length(t_list)
    t_sample_list = t0:nsample*dt:tf; Nt_sample_list = length(t_sample_list)
    mean_L1 = Array{Float64,1}(Nt_list)
    mean_L2 = similar(mean_L1)
    variance_L1 = similar(mean_L1)
    variance_L2 = similar(mean_L1)
    statesample_list = Array{StateMatrix,1}(Nt_sample_list)

    C_array = circshift(Cm0,[m_min,m_min])
    # Do the computation for the initial state
    statesample_list[1] = copy(Cm0)
    recordmeanvariance!(C_array,mean_L1,mean_L2,variance_L1,variance_L2,1)
    # Do the loop
    for ind in 2:Nt_list
        # C_array = UKdt_half.*C_array
        C_array_temp = similar(C_array)
        broadcast!(*,C_array_temp,UKdt_half,C_array)
        T_xp! * C_array_temp
        broadcast!(*,C_array,UVdt, C_array_temp)
        T_px! * C_array
        C_array = UKdt_half .* C_array
        # mean and variance
        recordmeanvariance!(C_array,mean_L1,mean_L2,variance_L1,variance_L2,ind)
        # sample if necessary
        if mod(ind-1,nsample)==0
            ind_sample = div(ind-1,nsample) + 1
            statesampling!(C_array,statesample_list,ind_sample,m_min)
        end
    end
    return t_list,mean_L1,mean_L2,variance_L1,variance_L2,t_sample_list,statesample_list,circshift(C_array,[-m_min,-m_min])
end

"""
    svector2statematrix!(mp::Int, psi_s::Array, ms_min::Int, ms_max::Int, m_min::Int, m_max::Int) --> Cm12::StateMatrix

    Given the value 'mp', this function converts a vector 'psi_s' in the Hs space into a state matrix, and fill it to Cm12.

    The returned value 'Cm12' is an Npoints×Npoints matrix, where Npoints = m_max - m_min + 1. The length of psi_s should be ms_max-ms_min+1.

    Caution: This function requires that mp is compatible to the psi_s vector in terms of the combined condition.
    An error will be incurred if the requirement is not satisfied.
"""
function svector2statematrix(mp::Int, psi_s::Array{T,1}, ms_min::Int, ms_max::Int, m_min::Int, m_max::Int) where T<:Number
    Npoints = m_max - m_min + 1
    Ns = ms_max - ms_min + 1
    Cm12 = zeros(Complex128,Npoints,Npoints)
    for ind = 1:Ns
        ms = ind + ms_min - 1
        if mod(ms+mp,2)!=0
            @assert abs(psi_s[ind])<1e-12
        else
            m1 = div(mp+ms,2); m2 = div(mp-ms,2);
            ind1 = m1 - m_min + 1
            ind2 = m2 - m_min + 1
            if m1>m_min && m1<m_max && m2>m_min && m2<m_max
                Cm12[ind1,ind2] = psi_s[ind]
            end
        end
    end
    return Cm12
end

svector2statematrix(mp::Int, psi_s::Array{T,1}, m_min::Int, m_max::Int) where T<:Number =
     svector2statematrix(mp,psi_s,m_min,m_max,m_min,m_max)


"""
    project_nth_odd_subspace(Cm12::StateMatrix, n::Int, mp::Int)

Projection of Cm12 to the odd-ms subspace of the nth energy band.

Note that for every odd subspace, ms can only takes odd values. Therefore, if mp
is even, then the projections is set to be 0.
"""
function project_nth_odd_subspace(Cm12::StateMatrix, n::Int, mp::Int)
    if mod(mp,2)==0; return zeros(Cm12); end
    i1 = 4*(n-1) + 2
    i2 = 4*(n-1) + 3
    Cm_n1 = svector2statematrix(mp,Vec_Hs[:,i1],m_min,m_max)
    Cm_n2 = svector2statematrix(mp,Vec_Hs[:,i2],m_min,m_max)
    Cm_out = sum(conj.(Cm_n1).*Cm12)*Cm_n1 + sum(conj.(Cm_n2).*Cm12)*Cm_n2
    return Cm_out
end

function Ls_longtime_average(Cm12::StateMatrix, m_min::Int, m_max::Int)
    result = 0.0
    proj_accumulation = 0.0
    Npoints = m_max - m_min + 1
    for n = 1:Nbands
        for mp = m_min:m_max
            iseven(mp) && continue
            C_proj = project_nth_odd_subspace(Cm12,n,mp)
            C_proj_array = circshift(C_proj,[m_min,m_min])
            C2 = abs2.(C_proj_array)
            result += sum(C2.*Ls_ope)
            proj_accumulation += sum(C2)
        end
        if abs(1.0-proj_accumulation)<1e-9
            break
        end
    end
    return result
end

"""
                function kicking_evolving_averaging

function kicking_evolving_averaging(Cm0::StateMatrix,t0::Real,tf::Real,dt::Real,
                                    k_tot::Int,k_step::Int,n_kicking::Int,
                                    m_min::Int,m_max::Int)

'n_kicking' determines how many dt steps it waits before the next kicking.
For now, let's not do sampling along the way.
"""
function kicking_evolving_averaging(Cm0::StateMatrix,t0::Real,tf::Real,dt::Real,
                                    k_tot::Int,k_step::Int,n_kicking::Int,
                                    m_min::Int,m_max::Int)
    dt_kicking = dt * n_kicking
    number_kicking = div(k_tot,k_step)
    k_remainder = mod(k_tot,k_step)
    Cm_current = Cm0
    Cm_kicked = similar(Cm_current)
    #define arrays to store time scales
    t_list = Float64[]
    t_kicked = Float64[]
    # t_sample_list = Float64[] Let's not do sampling for now
    t = 0.0
    # define arrays to store data
    mean_L1 = Float64[]
    mean_L2 = Float64[]
    variance_L1 = Float64[]
    variance_L2 = Float64[]
    # statesample_list = StateMatrix[] Let's not do sampling for now

    for i_kicking = 1:number_kicking
        t0_stage = t
        tf_stage = t + dt_kicking
        circshift!(Cm_kicked,Cm_current,[k_step,0])
        push!(t_kicked,t)
        tl,mL1,mL2,vL1,vL2,tsl,ssl,Cm_current = trotter_suzuki_solver(Cm_kicked,m_min,m_max,t0_stage,tf_stage,dt,100000)
        append!(mean_L1,mL1[1:end-1])
        append!(mean_L2,mL2[1:end-1])
        append!(variance_L1,vL1[1:end-1])
        append!(variance_L2,vL2[1:end-1])
        append!(t_list,tl[1:end-1])
        t = tl[end]
    end
    circshift!(Cm_kicked,Cm_current,[k_remainder,0])
    t0_stage = t
    tf_stage = tf
    if k_remainder>0
        push!(t_kicked,t)
    end
    tl,mL1,mL2,vL1,vL2,tsl,ssl,Cm_current = trotter_suzuki_solver(Cm_kicked,m_min,m_max,t0_stage,tf_stage,dt,100000)
    append!(mean_L1,mL1[1:end])
    append!(mean_L2,mL2[1:end])
    append!(variance_L1,vL1[1:end])
    append!(variance_L2,vL2[1:end])
    append!(t_list,tl[1:end])
    Lslta = Ls_longtime_average(Cm_current,m_min,m_max)
    Lplta = k_tot
    L2lta = 0.5*(Lplta - Lslta)
    transeff = L2lta / k_tot
    return t_list,t_kicked,mean_L1,mean_L2,variance_L1,variance_L2,transeff,Cm_current
end

# functions previously belonging to "ergotropy.jl" file.
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
