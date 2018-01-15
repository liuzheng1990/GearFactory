module GearToolkit

export trajectorystatistics, momentumplot

include("includes.jl")
import JLD: save, load
import Plots: plot, plot!, savefig, title!,gr

function trajectorystatistics(traj::GearTrajectory_Ket, strFile::AbstractString="nosave")
	psi_t = traj.psi_t
	tout = traj.tout

	NDimension1 = traj.glab.env.NDimension1
	NDimension2 = traj.glab.env.NDimension2
	L1 = traj.glab.L1
	L2 = traj.glab.L2
	w_d = traj.glab.env.w_d

	mean_L1 = zeros(Float64,length(tout))
	mean_L2 = zeros(Float64,length(tout))
	std_L1 = zeros(Float64,length(tout))
	std_L2 = zeros(Float64,length(tout))

	println("statistics:computing mean and std...")
	for i=1:length(tout)
		temp=psi_t[i] #temp is the state ket.
		rho1=partialtrace2(temp.data, NDimension1, NDimension2)
		rho2=partialtrace1(temp.data, NDimension1, NDimension2)
		mean_L1[i] = expect_dm(rho1, L1)
		mean_L2[i] = expect_dm(rho2, L2)
		std_L1[i] = sqrt(variance_dm(rho1, L1))
		std_L2[i] = sqrt(variance_dm(rho2, L2))
	end
	if strFile != "nosave"
		if strFile == ""
			strFile = "statistics($(traj.glab.env.alpha),$(traj.glab.env.w_d),$(traj.glab.env.V_0),$(traj.glab.env.V_d)).jld"
		end
		save(strFile,"env",traj.glab.env,"mean_L1",mean_L1,"mean_L2",mean_L2,"std_L1",std_L1,"std_L2",std_L2)
	end
	return mean_L1, mean_L2, std_L1, std_L2
end #function

function momentumplot(env::EnvSpec, tout::Array, mean_L1::Array, mean_L2::Array, std_L1::Array, std_L2::Array, strFile::AbstractString="")
	#println("momentum-plot:loading...")
#	psi_t = traj.psi_t
#	tout = traj.tout

#	NDimension1 = traj.glab.env.NDimension1
#	NDimension2 = traj.glab.env.NDimension2
#	L1 = traj.glab.L1
#	L2 = traj.glab.L2
	w_d = env.w_d
#
#	mean_L1 = zeros(Float64,length(tout))
#	mean_L2 = zeros(Float64,length(tout))
#	std_L1 = zeros(Float64,length(tout))
#	std_L2 = zeros(Float64,length(tout))
#
#	println("momentum-plot:computing mean and std...")
#	for i=1:length(tout)
#		temp=psi_t[i] #temp is the state ket.
#		rho1=partialtrace2(temp.data, NDimension1, NDimension2)
#		rho2=partialtrace1(temp.data, NDimension1, NDimension2)
#		mean_L1[i] = expect_dm(rho1, L1)
#		mean_L2[i] = expect_dm(rho2, L2)
#		std_L1[i] = sqrt(variance_dm(rho1, L1))
#		std_L2[i] = sqrt(variance_dm(rho2, L2))
#	end
	plot(tout,mean_L1,label="mean_L:free gear",color=:black)
	plot!(tout,mean_L1+std_L1,label="std_L:free gear",color=:blue)
	plot!(tout,mean_L1-std_L1,color=:blue)
	plot!(tout,mean_L2,label="mean_L:driving gear",color=:red)
	plot!(tout,mean_L2+std_L2,label="std_L:driving gear",color=:yellow)
	plot!(tout,mean_L2-std_L2,color=:yellow)
	plot!(tout,w_d*TOF.(env, tout),label="external field")
	strTitle = "alpha=$(env.alpha), w_d=$(w_d), V_0=$(env.V_0), V_d=$(env.V_d)"
	title!(strTitle)
	if strFile == ""
		strFile = strTitle
	end
	savefig(strFile)
	println("momentum-plot:Done!")
end #function

function momentumplot(traj::GearTrajectory_Ket, mean_L1::Array, mean_L2::Array, std_L1::Array, std_L2::Array, strFile::AbstractString="")
	return momentumplot(traj.glab.env, traj.tout,mean_L1,mean_L2,std_L1,std_L2,strFile)
end #function

end #module
