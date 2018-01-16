"""
This file atempts to implement the class "GearLab".

"GearLab" stores the useful operators and pointers to different quantum - gear
objects.

In the end, it should be the only manager of the operators, but temporarily, it
may need to update the operators to the global namespace. These updating function
will hopefully be deprecated, together with the global namespace.
"""
module GearLaboratory

export GearLab, update_GearLab
export b1,b2,b,L1,L2,L,K1,K2,K,H_d,externalH,H,one_gear_stationary_hamiltonian

using LibrariesFunctions,QuantumOptics,EnvironmentSpecification

mutable struct GearLab #consider immutable
    env::EnvSpec
    b1::Basis; b2::Basis; b::Basis
    L1::Operator; L2::Operator; L::Operator
    K1::Operator; K2::Operator; K::Operator
    H_d::Operator

    function GearLab(env::EnvSpec)
        #unpack env specifications
        alpha = env.alpha
        w_d = env.w_d
        V_0 = env.V_0
        V_d = env.V_d
		I_1 = env.I_1
		I_2 = env.I_2
        m1_min = env.m1_min
        m1_max = env.m1_max
        m2_min = env.m2_min
        m2_max = env.m2_max
        NDimension1 = env.NDimension1
        NDimension2 = env.NDimension2

        #construct the bases and the operators
        b1 = NLevelBasis(NDimension1)
        b2 = NLevelBasis(NDimension2)
        b = b1⊗b2

        L1 = angular_momentum_operator(b1,m1_min,NDimension1)
        L2 = angular_momentum_operator(b2,m2_min,NDimension2)
        L = embed(b,1,L1) + embed(b,2,L2)

        K1 = kinetic_operator(b1,m1_min,NDimension1,I_1)
        K2 = kinetic_operator(b2,m2_min,NDimension2,I_2)
        K = embed(b,1,K1) + embed(b,2,K2)

        Raise1 = raising_operator(b1,NDimension1)
        Raise2 = raising_operator(b2,NDimension2)
        Lower1 = dagger(Raise1)
        Lower2 = dagger(Raise2)
        H_d = V_d/2*(Raise1⊗Lower2 + Lower1⊗Raise2)
        new(env, b1,b2,b, L1,L2,L, K1,K2,K, H_d)
    end #function
end #mutable struct

function externalH(glab::GearLab, t::Real)
    r2 = raising_operator(glab.b2, glab.env.NDimension2, 2)
    M = exp(-im*2*θₚ(glab.env,t))*r2
    M = 1/2 * (M + dagger(M)) + identityoperator(glab.b2)
    return -glab.env.V_0/2 * embed(glab.b,2,M)
end #function

function H(glab::GearLab, t::Real, psi)
    return glab.K + glab.H_d + externalH(glab, t)
end #function

"""
function one_gear_stationary_hamiltonian(glab::GearLab)


returns the hamiltonian of the driving gear alone, in the stationary external potential.

"""

function one_gear_stationary_hamiltonian(glab::GearLab)
	M = raising_operator(glab.b2, glab.env.NDimension2,2)
	M = 1/2 * (M + dagger(M)) + identityoperator(glab.b2)
	H_e = glab.K2 - glab.env.V_0/2 * M
	return H_e
end #function


function update_GearLab(glab::GearLab)
    global b1,b2,b,L1,L2,L,K1,K2,K,H_d,externalH,H
    b1 = glab.b1
    b2 = glab.b2
    b = glab.b
    L1 = glab.L1
    L2 = glab.L2
    L = glab.L
    K1 = glab.K1
    K2 = glab.K2
    K = glab.K
    H_d = glab.H_d
	
	externalH(t::Real) = externalH(glab,t)
	H(t::Real, psi) = H(glab,t,psi)
    
end #update_GearLab



end #module
