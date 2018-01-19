module EnvironmentSpecification

export EnvSpec, update_EnvSpec
export alpha,w_d,V_0,V_d,I_1,I_2,m1_min,m1_max,m2_min,m2_max,NDimension1,NDimension2,θₚ,TOF

struct EnvSpec
    alpha::Float64
    w_d::Float64
    V_0::Float64
    V_d::Float64
    I_1::Float64
    I_2::Float64
    m1_min::Int64
    m1_max::Int64
    m2_min::Int64
    m2_max::Int64

    NDimension1::Int64
    NDimension2::Int64
    EnvSpec(alpha::Real, w_d::Real, V_0::Real, V_d::Real, I_1::Real, I_2::Real, m1_min::Int, m1_max::Int, m2_min::Int, m2_max::Int) = new(alpha, w_d, V_0, V_d, I_1, I_2, m1_min, m1_max, m2_min, m2_max, m1_max - m1_min + 1, m2_max - m2_min + 1)
end #struct

"""
EnvSpec(alpha::Real, w_d::Real, V_0::Real, V_d::Real, I_1::Real, I_2::Real, m1_min::Int, m1_max::Int, m2_min::Int, m2_max::Int)

If m2_min and m2_max are not given, they are set to be m1_min and m1_max by default.
"""
EnvSpec(alpha::Real, w_d::Real, V_0::Real, V_d::Real, I_1::Real, I_2::Real, m1_min::Int, m1_max::Int) = EnvSpec(alpha, w_d, V_0, V_d, I_1, I_2, m1_min, m1_max, m1_min, m1_max)
#

θₚ(env::EnvSpec, t::Real) = env.w_d*(t+1/env.alpha*((env.alpha*t+2)*exp(-env.alpha*t)-2))
TOF(env::EnvSpec, t::Real) = 1 - (env.alpha*t+1)*exp(-env.alpha*t)
"""
update_EnvSpec(env::EnvSpec)

This function updates the value of the global variables, including the phase and the turning-on functions.
"""
function update_EnvSpec(env::EnvSpec)
    global alpha,w_d,V_0,V_d,I_1,I_2,m1_min,m1_max,m2_min,m2_max,NDimension1,NDimension2
    global θₚ, TOF

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
    #θₚ(t) = w_d*(t+1/alpha*((alpha*t+2)*exp(-alpha*t)-2))
    #TOF(t) = 1 - (alpha*t+1)*exp(-alpha*t)
	θₚ(t) = θₚ(env, t)
	TOF(t) = TOF(env,t)
end #function


end #module
