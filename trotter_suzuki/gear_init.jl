using QuantumOptics

const StateMatrix = Array{Complex{Float64},2}
const OperatorMatrix = Array{Complex{Float64},2}

include("gear_init_params.jl")

n = n1+n2; ns = Int64((n1^2+n2^2)/gcd(n1,n2));
p1 = div(n1,gcd(n1,n2)); p2 = div(n2,gcd(n1,n2))

Npoints = m_max - m_min + 1
m_list = m_min:m_max
mpoints_gear = m_list
dx = 2*pi/Npoints; x_list = range(0,dx,Npoints);

# I1 = rI; I2 = rI;
Ip = n^2*I1/(n1^2+n2^2); Is = Ip;
Ir = Is; Ic = Ip;
prd = ns*2*pi/n  # period of theta_r coordinate

## "gear_init_params.jl" should contain the values of n1 and n2

# dt = 0.01;
################################################################################

#  qojulia operators in Hs space

################################################################################
bs_qo = NLevelBasis(Npoints)
Ls_qo = SparseMatrixCSC(diagm(m_min:m_max))
Ls_qo = SparseOperator(bs_qo,bs_qo,Ls_qo)
arr_kick_minus = diagm(ones(Npoints-1),1)
kick_minus_qo = SparseOperator(bs_qo,bs_qo,arr_kick_minus)
arr_kick_plus = arr_kick_minus.'
kick_plus_qo = dagger(kick_minus_qo)

Ks_qo = Ls_qo^2/(2*Is)
Vs_qo = -V0/4*(kick_plus_qo^2 + kick_minus_qo^2)^2
Hs_qo = Ks_qo + Vs_qo
arr_Hs = Hermitian(full(Hs_qo.data))
D_Hs,Vec_Hs = eig(arr_Hs)

Nbands = div(length(D_Hs),4)

################################################################################

#  Trotter-Suzuki operators in H1⊗H2 space

################################################################################


m_list_shifted = circshift(m_list,m_min)    # {0,1,...,m_max,m_min,...,-2,-1}
x_boundary = findfirst(x->(x>pi),x_list)
x_list_shifted = circshift(x_list,-x_boundary+1)
x_list_shifted = mod2pi.(x_list_shifted.+pi).-pi    #[-pi,pi)

# Prepare the fft operators.
v = zeros(Complex128, Npoints, Npoints)
T_xp! = Npoints^2/(2*pi)*plan_ifft!(v,[1,2],flags=FFTW.MEASURE)
T_px! = 2*pi/Npoints^2*plan_fft!(v,[1,2],flags=FFTW.MEASURE)
T_xp = Npoints^2/(2*pi)*plan_ifft(v,[1,2],flags=FFTW.MEASURE)
T_px = 2*pi/Npoints^2*plan_fft(v,[1,2],flags=FFTW.MEASURE)

# prepare operators
L1_ope = [m_list_shifted[i] for i=1:Npoints,j=1:Npoints]
L1square_ope = L1_ope.^2
L2_ope = [m_list_shifted[j] for i=1:Npoints,j=1:Npoints]
L2square_ope = L2_ope.^2
K_ope = L1square_ope/(2*I1)+L2square_ope/(2*I2)
V_ope = [-V0/2*(cos(n1*x_list[i]-n2*x_list[j])+1) for i=1:Npoints,j=1:Npoints]
Ls_ope = n*n1/(n1^2+n2^2)*L1_ope - n*n2/(n1^2+n2^2)*L2_ope
Lp_ope = n*n2/(n1^2+n2^2)*L1_ope + n*n1/(n1^2+n2^2)*L2_ope

# functions
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


println("Variables and functions ready to use!")
