module TimeEvolution

export GearTrajectory, GearTrajectory_Ket, tGroundState, tTimeEvolution

using QuantumOptics, EnvironmentSpecification, GearLaboratory, LibrariesFunctions
#import JLD

abstract type GearTrajectory end

mutable struct GearTrajectory_Ket <: GearTrajectory
    glab::GearLab
    initial_state::Ket
    tout::Array{Float64,1}
    psi_t::Array{QuantumOptics.states.Ket,1}
    GearTrajectory_Ket(glab::GearLab,psi_0::Ket,tout::Array,psi_t::Any) = new(glab,psi_0,tout,psi_t)
end #struct

#constructor for a premature traj.
GearTrajectory_Ket(glab::GearLab,psi_0::Ket,tout::Array) = GearTrajectory_Ket(glab,psi_0,tout,Ket[])

#constructor from an old-fasioned trajectory file. Designed only for backward compatibility consideration.
function GearTrajectory_Ket(glab::GearLab, psi_t_file::AbstractString) #used to load old data
    d = JLD.load(psi_t_file)
    GearTrajectory(glab,d["psi_0"],d["tout"],d["psi_t"],psi_t_file)
end #function


"""
tGroundState

This function computes the ground state of Gear 2 in the external field, and construct a
tensor product state, with Gear 1's state to be of the same shape of Gear2, but shifted by pi.

This function requires m1_min == m2_min and m1_max == m2_max.
"""
function tGroundState(V_0::Real, b1::Basis, b2::Basis, m2_min::Int, NDimension2::Int, K2::Operator)
    M = raising_operator(b2,NDimension2,2)
	M = 1/2 * (M + dagger(M)) + identityoperator(b2)
	HH = K2 - V_0/2 * M

	H_matrix = Hermitian(full(HH.data))
	D,V = eig(H_matrix,-V_0,0)
	v1 = V[:,1]
	v2 = V[:,2]

	v_plus = Ket(b2,v1+v2)
	v_minus = Ket(b2,v1-v2)
	v_plus = v_plus/norm(v_plus)
	v_minus = v_minus/norm(v_minus)
	u_plus = zeros(length(v_plus.data))	#assemble the shifted state for the free gear.
	for i in 1:length(v_plus.data)
		m = get_m(i,m2_min)
		if m%2 != 0
		    sgn = -1
		else
		    sgn = 1
		end
		u_plus[i] = sgn * v_plus.data[i]
	end

	# x = -pi:0.1:pi
	# psi = get_wavefunction(x,v_plus.data,m2_min,m2_max)
	# psi2 = get_wavefunction(x,u_plus,m2_min,m2_max)
	# plot(x,abs2.(psi))
	# plot!(x,abs2.(psi2))

	u_plus = Ket(b1, u_plus)
	v0 = u_plus ⊗ v_plus
	#HH = H(0,0)
end #functiont

tGroundState(glab::GearLab) = tGroundState(glab.env.V_0, glab.b1, glab.b2, glab.env.m2_min, glab.env.NDimension2, glab.K2)

"""
tTimeEvolution

Compute the time evolution given a traj object and a Hamiltonian. It will call the
"schroedinger" or "schroedinger_dynamic" solvers depending on whether H given is an
operator or a function. The dispatch without H given calls the H function given in
"GearLaboratory.jl" by default.

function tTimeEvolution(traj::GearTrajectory_Ket, H::Operator)
function tTimeEvolution(traj::GearTrajectory_Ket, H::Function)
function tTimeEvolution(traj::GearTrajectory_Ket)
"""

function tTimeEvolution(traj::GearTrajectory_Ket, H::Operator)
	traj.tout,traj.psi_t = timeevolution.schroedinger(traj.tout,traj.initial_state,H)
    return 0
end #function

function tTimeEvolution(traj::GearTrajectory_Ket, H::Function)
    traj.tout,traj.psi_t = timeevolution.schroedinger_dynamic(traj.tout,traj.initial_state,H)
    return 0
end #function

function tTimeEvolution(traj::GearTrajectory_Ket)
    myH(t::Real, psi) = H(traj.glab, t, psi)
    tTimeEvolution(traj, myH)
end #function

end #module
