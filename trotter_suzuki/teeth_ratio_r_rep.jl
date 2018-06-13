using QuantumOptics
using Plots

n1 = 2; n2 = 4;
Npoints = 240;
I1 = 1.0;
V0 = 10.0;

#############################################################
n = n1+n2; ns = (n1^2+n2^2)/gcd(n1,n2);
Ir = n^2*I1/(n1^2+n2^2)
prd = ns*2*pi/n  # period of theta_r coordinate


b_position = PositionBasis(0.0,prd,Npoints)
b_momentum = MomentumBasis(b_position)
L = momentum(b_momentum)
x = position(b_momentum)
arr_L = L.data
arr_L[2,2]-arr_L[1,1]
exp_nr = expm(im*n*x)
KE = L^2/(2*Ir)
PE = -V0/2*((exp_nr+dagger(exp_nr))/2+identityoperator(b_momentum))
Hr = KE + PE
T_px = transform(b_momentum,b_position)
T_xp = dagger(T_px)
x_points = samplepoints(b_position)
p_points = samplepoints(b_momentum)

arr_Hr = Hermitian(Hr.data)
D_r, Vec_r = eig(arr_Hr)
x_axis = 1:100
plot(D_r[1:20])
# xticks!(1:1:100)
free_energy = ((x_axis.-1)/2*n/ns).^2/(2*Ir).-V0
plot(x_axis,free_energy)


vg = Ket(b_momentum, Vec_r[:,1])
plot(p_points,abs2.(vg.data))

plot(D_r[1:30],"*")

for i = 1:10
    vi = Vec_r[:,i]
    plot(p_points, real.(vi))
    # xticks(p_points)
    title!("energy level: $(i)")
    savefig("el_$(i).png")
end
