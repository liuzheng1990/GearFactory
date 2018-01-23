module GearToolkit

export trajectorystatistics, momentumplot, plotLdist, plotxdist

using LibrariesFunctions
using EnvironmentSpecification
using GearLaboratory
using QuantumOptics
using TimeEvolution
import JLD: save, load
import Plots: plot, plot!, savefig, title!, gr, vline!

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

function momentumplot(env::EnvSpec, tout::Union{Range,Array}, mean_L1::Array, mean_L2::Array, std_L1::Array, std_L2::Array, strFile::AbstractString="")
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

"""
plotLdist(glab::GearLab, psi::Array, nIndex::Int, t::Real, strDir::AbstractString)

plot the wavepacket of psi in L rep. The two arguments "nIndex" and "t" are only used for labeling and naming the image file.

"strDir" specifies the path of the directory into which the image file should be saved. The filename is automatically generated, with the help of "nIndex" and "t" arguments.

plotLdist(traj::GearTrajectory_Ket, dt_plot::Real, t_max_plot::Real, strDir::AbstractString)

If one do not wish to specify the max value of t, assign a negative number or a very large number to "t_max_plot".
"""
function plotLdist(glab::GearLab, psi::Array, nIndex::Int, t::Real, strDir::AbstractString)
	env = glab.env
	m1_min = env.m1_min; m1_max = env.m1_max;
	m2_min = env.m2_min; m2_max = env.m2_max;

	rho1 = partialtrace2(psi,env.NDimension1,env.NDimension2)
	p1 = real.(diag(rho1))
	rho2 = partialtrace1(psi,env.NDimension1,env.NDimension2)
	p2 = real.(diag(rho2))
	rho1 = 0
	rho2 = 0

	plot(m1_min:m1_max, p1, label="free gear")
	plot!(m2_min:m2_max, p2, label="driving gear")
	vline!([env.w_d])
	title!("$(env.alpha),$(env.w_d),$(env.V_0),$(env.V_d);t=$t")
	savefig("$(strDir)/$(nIndex):$(env.alpha),$(env.w_d),$(env.V_0),$(env.V_d);t=$t.png")
end #function

function plotLdist(traj::GearTrajectory_Ket, dt_plot::Real, t_max_plot::Real, strDir::AbstractString)
	glab = traj.glab
	tout = traj.tout
	n_step = findfirst(x->(x>=dt_plot), tout) - 1
	n_end = findfirst(x->(x>t_max_plot), tout)
	if n_end == 0 || t_max_plot < 0
		t_plot = tout[1:n_step:end]
		psi_t_plot = traj.psi_t[1:n_step:end]
	else
		t_plot = tout[1:n_step:(n_end-1)]
		psi_t_plot = traj.psi_t[1:n_step:(n_end-1)]
	end
	for i in 1:length(t_plot)
		t = t_plot[i]
		psi = psi_t_plot[i].data
		plotLdist(glab, psi, i, t, strDir)
	end
end #function

"""
plotxdist(glab::GearLab, x_list::Union{Range,Array}, psi::Array, nIndex::Int, t::Real, strDir::AbstractString)

plotxdist(traj::GearTrajectory_Ket, x_list::Union{Range,Array}, dt_plot::Real, t_max_plot::Real, strDir::AbstractString)

If you do not want to specify the end of the time array but want to plot till the end, assign a negative number or a very large number to "t_max_plot".
"""
function plotxdist(glab::GearLab, x_list::Union{Range,Array}, psi::Array, nIndex::Int, t::Real, strDir::AbstractString)
	env = glab.env

	rho1 = partialtrace2(psi,env.NDimension1,env.NDimension2)
	x1 = x_distribution_dm(x_list, rho1, env.m1_min, env.m1_max)
	rho1 = 0
	rho2 = partialtrace1(psi,env.NDimension1,env.NDimension2)
	x2 = x_distribution_dm(x_list, rho2, env.m2_min, env.m2_max)
	rho2 = 0

	plot(x_list, x1, label="free gear")
	plot!(x_list, x2, label="driving gear")
	title!("x: $(env.alpha),$(env.w_d),$(env.V_0),$(env.V_d);t=$t")
	savefig("$(strDir)/$(nIndex):$(env.alpha),$(env.w_d),$(env.V_0),$(env.V_d);t=$t.png")
end #function


function plotxdist(traj::GearTrajectory_Ket, x_list::Union{Range,Array}, dt_plot::Real, t_max_plot::Real, strDir::AbstractString)
	glab = traj.glab
	tout = traj.tout
	n_step = findfirst(x->(x>=dt_plot), tout) - 1
	n_end = findfirst(x->(x>t_max_plot), tout)
	if n_end == 0 || t_max_plot < 0
		t_plot = tout[1:n_step:end]
		psi_t_plot = traj.psi_t[1:n_step:end]
	else
		t_plot = tout[1:n_step:(n_end-1)]
		psi_t_plot = traj.psi_t[1:n_step:(n_end-1)]
	end
	for i in 1:length(t_plot)
		t = t_plot[i]
		psi = psi_t_plot[i].data
		plotxdist(glab, x_list, psi, i, t, strDir)
	end
end #function
end #module
