module ClassicalInitial

using EnvironmentSpecification
using GearLaboratory
using TimeEvolution
using QuantumOptics
using LibrariesFunctions

export ClassicalInit, generate_initial_data

struct ClassicalInit
    env::EnvSpec
    Npoints::Int64
    mean_x1::Float64; std_x1::Float64; mean_L1::Float64; std_L1::Float64;
    mean_x2::Float64; std_x2::Float64; mean_L2::Float64; std_L2::Float64;
end #struct

function ClassicalInit(glab::GearLab, psi_0::Array, Npoints::Int)
    x_list = -pi:0.001:pi
    rho1 = partialtrace2(psi_0,glab.env.NDimension1,glab.env.NDimension2)
    mean_L1 = expect_dm(rho1,glab.L1)
    std_L1 = sqrt(variance_dm(rho1,glab.L1))
    px1 = x_distribution_dm(x_list, rho1, glab.env.m1_min, glab.env.m1_max)

	#Since in this problem px1 is large at -pi and pi, I need to shift it by pi
	#in order to compute the mean value. 
	c_negative = count(x->(x<0), x_list)
	px1_shifted = px1[c_negative+1:end]
	append!(px1_shifted, px1[1:c_negative])
	####

    mean_x1 = simps(px1_shifted.*x_list, 0.001)
    std_x1 = sqrt(simps(px1_shifted.*x_list.^2, 0.001)-mean_x1^2)
	mean_x1 += pi

    #rho1 = 0
    rho2 = partialtrace1(psi_0,glab.env.NDimension1,glab.env.NDimension2)
    mean_L2 = expect_dm(rho2,glab.L2)
    std_L2 = sqrt(variance_dm(rho2,glab.L2))
    px2 = x_distribution_dm(x_list, rho2, glab.env.m2_min, glab.env.m2_max)
    mean_x2 = simps(px2.*x_list, 0.001)
    std_x2 = sqrt(simps(px2.*x_list.^2, 0.001)-mean_x2^2)
    #rho2 = 0
    return ClassicalInit(glab.env,Npoints,mean_x1,std_x1,mean_L1,std_L1,mean_x2,std_x2,mean_L2,std_L2)
end #function

ClassicalInit(traj::GearTrajectory_Ket, Npoints::Int) = ClassicalInit(traj.glab,traj.initial_state,Npoints)

function generate_initial_data(cinit::ClassicalInit)
    I_1 = cinit.env.I_1
    I_2 = cinit.env.I_2
    Npoints = cinit.Npoints
    x1_init = randn(Npoints).*cinit.std_x1 .+ cinit.mean_x1
    L1_init = randn(Npoints).*cinit.std_L1 .+ cinit.mean_L1
    x2_init = randn(Npoints).*cinit.std_x2 .+ cinit.mean_x2
    L2_init = randn(Npoints).*cinit.std_L2 .+ cinit.mean_L2
    initial_data = [[x1_init[i],x2_init[i],L1_init[i]/I_1,L2_init[i]/I_2] for i in 1:cinit.Npoints]
    return initial_data
end #function
end #module
