# using QuantumOptics

const StateMatrix = Array{Complex{Float64},2}
const OperatorMatrix = Array{Complex{Float64},2}

include("gear_init_params.jl")

# import functions
include("gear_init_functions.jl")

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
# bs_qo = NLevelBasis(Npoints)
# Ls_qo = SparseMatrixCSC(diagm(m_min:m_max))
# Ls_qo = SparseOperator(bs_qo,bs_qo,Ls_qo)
# arr_kick_minus = diagm(ones(Npoints-1),1)
# kick_minus_qo = SparseOperator(bs_qo,bs_qo,arr_kick_minus)
# arr_kick_plus = arr_kick_minus.'
# kick_plus_qo = dagger(kick_minus_qo)
#
# Ks_qo = Ls_qo^2/(2*Is)
# Vs_qo = -V0/4*(kick_plus_qo^2 + kick_minus_qo^2)^2
# Hs_qo = Ks_qo + Vs_qo
# arr_Hs = Hermitian(full(Hs_qo.data))
# D_Hs,Vec_Hs = eig(arr_Hs)
#
# Nbands = div(length(D_Hs),4)

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

################################################################################

#  Matrix-form operators in H1⊗H2 (kron) space

################################################################################

L_gear = spdiagm(Float64.(mpoints_gear))
raising_gear = sparse(diagm(ones(Float64,Npoints-1),-1))
lowering_gear = ctranspose(raising_gear)
identity_gear = speye(Float64,Npoints)
L_gear_square = spdiagm(Float64.(mpoints_gear.^2))

raising_gear_n1 = sparse(diagm(ones(Float64,Npoints-n1),-n1)) # This 4 operators are the exp(inθ₁) and exp(-inθ₁)
lowering_gear_n1 = ctranspose(raising_gear_n1)       # operators we need to define the cosine potential
raising_gear_n2 = sparse(diagm(ones(Float64,Npoints-n2),-n2))
lowering_gear_n2 = ctranspose(raising_gear_n2)


# Now let's set up the combined-space operators
KE = Hermitian(kron(L_gear.^2/(2*I1),identity_gear) + kron(identity_gear,L_gear.^2/(2*I2)))
cosine_V = Hermitian((kron(lowering_gear_n1,raising_gear_n2)+kron(raising_gear_n1,lowering_gear_n2))/2)
PE = Hermitian(-V0/2*(cosine_V+kron(identity_gear,identity_gear)))
H = Hermitian(full(KE + PE))

D_full,Vec_full = eig(H,-V0,0)

# Finally we haven seen that the ground-state energy of H matches that of H_r, meaning E_c = 0. We can check this explicitly, using Lc
vg = Vec_full[:,1]
Lc = n*n2/(n1^2+n2^2)*kron(L_gear,identity_gear) + n*n1/(n1^2+n2^2)*kron(identity_gear,L_gear)
Lc_square = Lc^2
dot(vg,Lc*vg)
dot(vg,Lc_square*vg)

# Aha! After a stupid mistake, we have seen that the ground state is indeed the 0'th eigenstate of L_c.
# Now it's time to assemble these state into the Trotter-Suzuki form and kick it!
# Note that the Trotter-Suzuki solver does all the computations in m1,m2 basis, and has no reference to the relative and com coordinates.

# Codes to assemble Cm given its rep in m1,m2.



Cm0 = v2Cm(vg,Npoints)




println("Variables and functions ready to use!")
