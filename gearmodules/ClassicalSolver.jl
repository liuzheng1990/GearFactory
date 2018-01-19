module ClassicalSolver

using OrdinaryDiffEq
using ClassicalInitial
import EnvironmentSpecification: EnvSpec, θₚ, TOF
import Base.Test: @test

export ClassicalSol, dynamics, solving_slave, ctimeevolution!

mutable struct ClassicalSol
    cinit::ClassicalInit
    tspan::Tuple{Float64,Float64}
    dt::Float64; dt_hist::Float64

#below are fields which can only be obtained after "ctimeevolution!" is run.     
	mean_traj::Array{Float64,2}
	std_traj::Array{Float64,2}
	hists::Array{Array{Float64,2},1}
	t_list
	i_hist
end #struct

"""
ClassicalSol(cinit::ClassicalInit, tspan::Float64, dt::Float64, dt_hist::Float64)

Fields like mean_traj, std_traj, hists, t_list, i_hist are set to empty. This constructor is used to load the data necessary to run the computation.
"""
ClassicalSol(cinit::ClassicalInit, tspan::Tuple, dt::Float64, dt_hist::Float64) = ClassicalSol(cinit,tspan,dt,dt_hist,Array{Float64}(0,0),Array{Float64}(0,0),Array{Float64,2}[],0,0)

function dynamics(env::EnvSpec, t::Real, u::Array, du::Array)
	V_0 = env.V_0; V_d = env.V_d
	I_1 = env.I_1; I_2 = env.I_2
	θₚ(t::Real) = θₚ(env, t)
    θ₁ = u[1]
    θ₂ = u[2]
    dθ₁ = u[3]
    dθ₂ = u[4]
    du[1] = dθ₁
    du[2] = dθ₂
    du[3] = -V_d/I_1 * sin(θ₂-θ₁)
    du[4] = V_d/I_2 * sin(θ₂-θ₁)-V_0/I_2 * sin(2*(θ₂-θₚ(t)))
end

function solving_slave(env::EnvSpec, initial_data::Array, tspan::Tuple, dt::Real, dt_hist::Real)
    N_data = length(initial_data)
    #println("N_data=$(N_data)")
    t_list = tspan[1]:dt:tspan[2]
    #trajectories = []
    mean_trajectory = zeros(Float64,4,length(t_list))
    std_trajectory = zeros(Float64,4,length(t_list))
    iStep = round(Int,dt_hist/dt)
    @test iStep>0
    i_hist = 1:iStep:length(t_list)
    hists = Array{Float64,2}[]
    #final_data = []
	mydynamics(t::Real, u::Array, du::Array) = dynamics(env,t,u,du)
    for u₀ in initial_data
        prob=ODEProblem(mydynamics,u₀,tspan)
        sol=solve(prob, Vern8(), abstol=1e-12, reltol=1e-12)
        sol_array = collect(sol(t_list))
        mean_trajectory = mean_trajectory .+ sol_array
        std_trajectory = std_trajectory .+ sol_array.^2
        push!(hists, sol_array[:,i_hist])
        #push!(final_data, sol_array[:,end])
    end
    #Now compute the average trajectory and the standard deviation
    mean_trajectory = mean_trajectory ./ N_data
    std_trajectory = sqrt.(abs.(std_trajectory./N_data .- mean_trajectory.^2))
    #return (final_data,mean_trajectory,std_trajectory,hists,t_list,i_hist)
	return (mean_trajectory,std_trajectory,hists,t_list,i_hist)
    #store the iterator i_hist used to slice t_list for t_hist
    #that is, to retrieve time _hist for hists, use:
    #t_hist = t_list[i_hist]
    #@test length(final_data) == length(t_list[i_hist])
end #function

function solving_slave(csol::ClassicalSol)
	initial_data = generate_initial_data(csol.cinit)
	return solving_slave(csol.cinit.env, initial_data, csol.tspan, csol.dt, csol.dt_hist)
end #function

function ctimeevolution!(csol::ClassicalSol)
	csol.mean_traj, csol.std_traj, csol.hists, csol.t_list, csol.i_hist = solving_slave(csol)
end #function

end #module
