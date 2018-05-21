include("gear_init.jl")
using JLD

vg = Vec_Hs[:,1];
Cm0 = svector2statematrix(0,vg,m_min,m_max)
t0 = 0.0; tf = 200.0; dt = 0.01;

k_step = 1;
n_kicking_list = 0:10:2000;
t_list_list = Float64[];
mean_L1_list = Float64[];
mean_L2_list = Float64[];
variance_L1_list = Float64[];
variance_L2_list = Float64[];
transeff_list = zeros(Float64,length(n_kicking_list))

function poissontimestep(rate::Real)
	u = rand()
	while abs(1-u)<1e-14
		u = rand()
	end
	x = -log(1-u)/rate
	return x
end


function poissonkicking(Cm0::StateMatrix, k_step::Int, rate::Real, t0::Real, tf::Real, dt::Real, nsample::Int,
						m_min::Int, m_max::Int)
	t_list= Float64[]
	t_sample_list = Float64[]
	mean_L1_list = Float64[]
	mean_L2_list = Float64[]
	variance_L1_list = Float64[]
	variance_L2_list = Float64[]
	statesample_list = StateMatrix[]

	t = t0
	dt_step = poissontimestep(rate)
	Cm_current = Cm0
	Cm_kicked = circshift(Cm0,[k_step,0])

	while t+dt_step < tf
		if dt_step < dt # save data for t, and just do a one-step evolution
			push!(t_list,t)
			m = mean_operator(Cm_kicked,L1_ope)
			v = mean_operator(Cm_kicked,L1square_ope)-m^2
			push!(mean_L1_list,m)
			push!(variance_L1_list,v)
			m = mean_operator(Cm_kicked,L2_ope)
			v = mean_operator(Cm_kicked,L2square_ope)-m^2
			push!(mean_L2_list,m)
			push!(variance_L2_list,v)
			push!(t_sample_list,t)
			push!(statesample_list,Cm_kicked)

			C_current = infinitesimal_trotter_suzuki_solver(Cm_kicked, m_min, m_max, dt_step)
		else
			tl,mean_L1,mean_L2,variance_L1,variance_L2,tsl,sl,Cm_current=trotter_suzuki_solver(Cm_kicked,m_min,m_max,t,t+dt_step,dt,nsample)
			append!(t_list,tl[1:end-1])
			append!(mean_L1_list,mean_L1[1:end-1])
			append!(mean_L2_list,mean_L2[1:end-1])
			append!(variance_L1_list,variance_L1[1:end-1])
			append!(variance_L2_list,variance_L2[1:end-1])
			append!(t_sample_list,tsl[1:end-1])
			append!(statesample_list,sl[1:end-1])
		end
		circshift!(Cm_kicked,Cm_current,[k_step,0])
		t += dt_step
		dt_step = poissontimestep(rate)
	end
	# note that the last step is not recorded into the list. I should either correct this, or argue that it doesn't matter
	return t_list, mean_L1_list, mean_L2_list, variance_L1_list, variance_L2_list, t_sample_list, statesample_list
end #function
