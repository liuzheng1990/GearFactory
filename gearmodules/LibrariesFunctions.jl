module LibrariesFunctions

export get_m,get_index,kinetic_energy,angular_momentum_operator,kinetic_operator,raising_operator
export wavefunction,get_wavefunction,c_k,boosted_state_reloaded,boosted_state
export partialtrace1,partialtrace2,densitymatrix_p2x,x_distribution_dm,expect_dm,variance_dm
export simps,dynamics

using QuantumOptics, Plots
#, JLD
import Base.Test: @test
################################################################################
function get_m(ind, m_min)
    return ind + m_min - 1
end

function get_index(m, m_min)
    return m - m_min + 1
end

function kinetic_energy(m::Int,rI::Real)
    return m*m/2/rI
end

function angular_momentum_operator(b, m_min, NDimension)
    arrL = sparse(eye(Float64,NDimension))
    for i in 1:NDimension
        m=get_m(i,m_min)
        arrL[i,i]=m
    end
    L = SparseOperator(b,b,arrL)
    return L
end

#construct the kinetic terms K1 & K2, diagonal in their coordinates
function kinetic_operator(b,m_min,NDimension,rI)
    arrK = sparse(eye(Float64,NDimension))
    for i in 1:NDimension
        m=get_m(i,m_min)
        arrK[i,i]=kinetic_energy(m,rI)
    end
    K = SparseOperator(b,b,arrK)
    return K
end

function raising_operator(b,NDimension::Int,raiseby::Int=1)
    arr = sparse(zeros(Float64,NDimension,NDimension))
    for i=1:(NDimension-raiseby)
        arr[i+raiseby,i] = 1
    end
    return SparseOperator(b,b,arr)
end

function wavefunction(x,coeffvector, m_min::Int, m_max::Int)
    result = 0.0+0.0*im
    for i in 1:(m_max-m_min+1)
        result += coeffvector[i]*exp(im*get_m(i,m_min)*x)/sqrt(2*pi)
    end
    return result
end

function get_wavefunction(x_list, coeffvector,m_min,m_max)
    psi = [wavefunction(x,coeffvector,m_min,m_max) for x in x_list]
    return psi
end

###############################################
#boosted functions
function c_k(coeffvector, k::Int, t::Real, rI::Real, m_min::Int, m_max::Int)
    summation = Complex{Float64}(0)
    #factor = exp(-im/2*rI*w^2*t)
    factor = exp(-im*k*θₚ(t))
    for i in 1:length(coeffvector)
        m = get_m(i,m_min)
        term1 = coeffvector[i]
        #term2 = exp(-im*m*w*t)
        term3 = sinc(m+rI*w_d*TOF(t)-k)
        summation += term1*term3
    end
    return factor*summation
end

function boosted_state_reloaded(coeffvector, t::Real, rI::Real, m_min::Int, m_max::Int, coeff_function)
    state = zeros(Complex{Float64},m_max-m_min+1)
    for i in 1:length(state)
        k=get_m(i,m_min)
        state[i]=coeff_function(coeffvector,k,t,rI,m_min,m_max)
    end
    return state
end

boosted_state(coeffvector, t::Real, rI::Real, m_min::Int, m_max::Int) = boosted_state_reloaded(coeffvector,t,rI,m_min,m_max,c_k)

function partialtrace1(arrState::Array, d1::Int, d2::Int)
    get_ind(m,n,d1,d2) = m+d1*(n-1)
    @test length(arrState)==d1*d2
    state = zeros(Complex{Float64},d2,d2)
    for n1 in 1:d2
        for n2 in 1:d2
            sum = Complex{Float64}(0)
            for m in 1:d1
                c_m_n1 = arrState[get_ind(m,n1,d1,d2)]
                c_m_n2 = arrState[get_ind(m,n2,d1,d2)]
                sum += c_m_n1 * conj(c_m_n2)
            end
            state[n1,n2] = sum
        end
    end
    return state
end

function partialtrace2(arrState::Array, d1::Int, d2::Int)
    get_ind(m,n,d1,d2) = m+d1*(n-1)
    @test length(arrState)==d1*d2
    state = zeros(Complex{Float64},d1,d1)
    for m1 in 1:d1
        for m2 in 1:d1
            sum = Complex{Float64}(0)
            for n in 1:d2
                c_m1_n = arrState[get_ind(m1,n,d1,d2)]
                c_m2_n = arrState[get_ind(m2,n,d1,d2)]
                sum += c_m1_n * conj(c_m2_n)
            end
            state[m1,m2] = sum
        end
    end
    return state
end

function densitymatrix_p2x(x_list, rho_p, m_min, m_max)   #a slow version
    rho_x = zeros(Complex{Float64},length(x_list),length(x_list))
    for i in 1:length(x_list)
        for j in 1:length(x_list)
            sum = Complex{Float64}(0)
            for m1 in m_min:m_max
                i1 = get_index(m1,m_min)
                for m2 in m_min:m_max
                    i2 = get_index(m2,m_min)
                    sum += rho_p[i1,i2]*exp(-im*(m1)*x_list[i])*exp(im*(m2)*x_list[j])
                end
            end
            rho_x[i,j] = sum
        end
    end
    return rho_x
end

function x_distribution_dm(x_list, rho_p, m_min, m_max) #only aim to obtain the x distribution
    p_x = zeros(Float64,length(x_list))
    for i in 1:length(x_list)
        sum = Float64(0)
        for m1 in m_min:m_max
            i1 = get_index(m1,m_min)
            for m2 in m_min:m_max
                i2 = get_index(m2,m_min)
                sum += rho_p[i1,i2]*exp(-im*(m2-m1)*x_list[i])
            end
        end
        p_x[i] = real(sum)
    end
    return p_x./(2*pi)
end

expect_dm(rho::Matrix, op::Union{Matrix,SparseMatrixCSC})=real(trace(rho*op))
expect_dm(rho::Matrix, op::Operator)=expect_dm(rho,op.data)
variance_dm(rho::Matrix, op::Union{Matrix,Operator})=expect_dm(rho,op*op)-expect_dm(rho,op)^2

function simps(y::Vector, h::Number)
    n = length(y)-1
    if n % 2 != 0
        return (simps(y[1:end-1],h)+simps(y[2:end],h))/2
    end
    s = sum(y[1:2:n] + 4*y[2:2:n] + y[3:2:n+1])
    return h/3 * s
end

function simps(x::Range, y::Vector)
    h = step(x)
    return simps(y,h)
end

function dynamics(t,u,du)
    θ₁ = u[1]
    θ₂ = u[2]
    dθ₁ = u[3]
    dθ₂ = u[4]
    du[1] = dθ₁
    du[2] = dθ₂
    du[3] = -V_d/I_1 * sin(θ₂-θ₁)
    du[4] = V_d/I_2 * sin(θ₂-θ₁)-V_0/I_2 * sin(2*(θ₂-θₚ(t)))
end
end #module
