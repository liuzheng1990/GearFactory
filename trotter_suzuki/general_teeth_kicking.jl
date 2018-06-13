include("gear_init.jl")
# include("ergotropy.jl")
# using QuantumOptics
using Plots
# import Plots: plot, plot!, title!, savefig
import JLD: save, @save, load, @load

# constants has been moved to "gear_init.jl" and "gear_init_params.jl" files.
# teeth_ratio folder (To do: update this file to GitHub)


# set up the bases and operators and transformations for one single gear.

"""
    single_kicking_data_collection

This function takes the n1,n2 value (defined globally) and does kickings to the
grond strength, by kicking strength from 1 to k_max. The kicking is done one single time
at t=0. The evolution is then simulated from 0 to tf. Frequency of state sampling is
set by the variable "nsample".

returned:
    t_list_list, t_sample_list_list,
    mean_L1_list, mean_L2_list, variance_L1_list, variance_L2_list,
    statesample_list_list

plotted:
    evolution of mean values for every kicking strength.
"""
function single_kicking_data_collection(Cm0::StateMatrix, k_min::Int, k_max::Int, tf::Real, nsample::Int)
    # prepare the containers for the returned values
    t_list_list = Array{Float64,1}[]
    t_sample_list_list = similar(t_list_list)
    mean_L1_list = similar(t_list_list)
    mean_L2_list = similar(mean_L1_list)
    variance_L1_list = similar(mean_L1_list)
    variance_L2_list = similar(mean_L1_list)
    statesample_list_list = Array{StateMatrix,1}[]

    for k = k_min:k_max
        println("k=$(k-k_min+1) out of $(k_max-k_min+1)...")
        Cm0_k = circshift(Cm0,[k,0])
        t_list,mean_L1,mean_L2,variance_L1,variance_L2,t_sample_list,statesample_list,Cmf=
                                trotter_suzuki_solver(Cm0_k, m_min, m_max, 0, tf, 0.01, nsample)
            # push the calculated data into the containers
        push!(t_list_list, t_list)
        push!(t_sample_list_list, t_sample_list)
        push!(mean_L1_list, mean_L1)
        push!(mean_L2_list, mean_L2)
        push!(variance_L1_list, variance_L1)
        push!(variance_L2_list, variance_L2)
        push!(statesample_list_list, statesample_list)

        # plot the mean values
        plot(t_list, mean_L1,label="mean: kicked gear")
        plot!(t_list, mean_L2, label="mean: driven gear")
        strtitle = "n1:$(n1); n2:$(n2); V0=$(V0); k=$k"
        title!(strtitle)
        savefig(strtitle * ".png")
    end
    return t_list_list, t_sample_list_list, mean_L1_list, mean_L2_list, variance_L1_list, variance_L2_list, statesample_list_list
end

# As a sanity check, we can do n1 = n2 = 2, V0 = 10, k_max = 10 and tf = 200.
# This should give us essentially the same result as we did previously using both solvers.
single_kicking_data_collection(Cm0, 10, 200, 1000)
# Beatifully agree. Now it's time to collect data for different gear ratios.


k_min = 3; k_max = 10
t_list_list, t_sample_list_list, mean_L1_list, mean_L2_list, variance_L1_list, variance_L2_list, statesample_list_list = single_kicking_data_collection(Cm0, k_min, k_max, 200, 10)
println("Computation Done! Saving data...")
save("data_n1=$(n1)_n2=$(n2)_V0=$(V0)_k_min=$(k_min)_k_max=$(k_max).jld","t_list_list",t_list_list, "t_sample_list_list",t_sample_list_list, "mean_L1_list", mean_L1_list, "mean_L2_list", mean_L2_list, "variance_L1_list", variance_L1_list,
            "variance_L2_list", variance_L2_list, "statesample_list_list", statesample_list_list)
println("data saved!")

# Let's compute the ergotropy of the sampled states and plot it together with the
# average KE of the second gear.

# release some unused memory first.
mean_L1_list=[];  variance_L1_list=[];
gc();
# include("ergotropy.jl")

ergo_list_list = Array{Float64,1}[]
KE2_list_list = Array{Float64,1}[]
KE2_prime_list_list = Array{Float64,1}[]
for ind = 1:(k_max-k_min+1)
    println("computing ergotropy: $(ind) out of $(k_max-k_min+1)...")
    t_list = t_sample_list_list[ind]
    statesample_list = statesample_list_list[ind]
    ergo_list = similar(t_list)
    KE2_list = similar(t_list)
    KE2_prime_list = similar(t_list)
    for i=1:length(t_list)
        rho2_t = rho2fromCm(statesample_list[i])
        ergo_list[i] = ergotropy(rho2_t,m_min,m_max)
        KE2_list[i] = trace(L_gear_square*rho2_t)/(2*I2)
        KE2_prime_list[i] = trace(L_gear*rho2_t)^2/(2*I2)
    end
    plot(t_list,ergo_list,label="ergotropy of Gear 2")
    plot!(t_list,KE2_list,label="average KE of Gear 2")
    plot!(t_list,KE2_prime_list,label="<L>^2/(2I_2)")
    title!("ergotropy and average KE: k=$(ind+k_min-1), n1=$n1, n2=$n2, V0=$V0")
    savefig("ergo_KE_Plot_k=$(ind+k_min-1)_n1=$(n1)_n2=$(n2).png")

    push!(ergo_list_list,ergo_list)
    push!(KE2_list_list,KE2_list)
    push!(KE2_prime_list_list,KE2_prime_list)
end

# a sanity chceck that all KE values are correct.
for ind = 1:(k_max-k_min+1)
    println("sanity check: plotting $(ind) out of $(k_max-k_min+1)...")
    t_list = t_list_list[ind]
    mean_L2 = mean_L2_list[ind]
    variance_L2 = variance_L2_list[ind]

    t_sample_list = t_sample_list_list[ind]
    KE2_list = KE2_list_list[ind]
    KE2_prime_list = KE2_prime_list_list[ind]

    # 1. check <L>^2 valus are correct.
    plot(t_sample_list,KE2_prime_list,label="<L>^2/(2I_2) using sampled states")
    plot!(t_list,mean_L2.^2/(2*I2),label="<L>^2/(2I_2) using mean_L2 lists")
    title!("sanity check:<L>^2/(2I),k=$(ind+k_min-1), n1=$n1, n2=$n2, V0=$V0")
    savefig("scheck_L2_k=$(ind+k_min-1)_n1=$(n1)_n2=$(n2).png")

    # 2. check KE2 by taking difference and comparing it with variance_L2
    plot(t_sample_list,KE2_list.-KE2_prime_list,label="difference in two types of KE averages.")
    plot!(t_list,variance_L2/(2*I2),label="variance of KE.")
    title!("sanity_check: KE and variance k=$(ind+k_min-1), n1=$n1, n2=$n2, V0=$V0")
    savefig("scheck_variance_k=$(ind+k_min-1)_n1=$(n1)_n2=$(n2).png")
end

# Let's plot out the ergotropy evolutions for each kicking
# for ind = 1:(k_max-k_min+1)
#     println("plotting ergotropy evolution: $(ind) out of $(k_max-k_min+1)...")
#     t_sample_list = t_sample_list_list[ind]
#     ergo_list = ergo_list_list[ind]
#     KE2_list = KE2_list_list[ind]
#
# end

# for ind = 1:(k_max-k_min+1)
#     println("plotting std(L2) evolution: $(ind) out of $(k_max-k_min+1)...")
#     t_list = t_list_list[ind]
#     std_L2_list = sqrt.(variance_L2_list[ind])
#     plot(t_list,std_L2_list)
#     title!("L2 fluctuation: k=$(ind+k_min-1), n1=$n1, n2=$n2, V0=$V0")
#     savefig("stdL2Plot_k=$(ind+k_min-1)_n1=$(n1)_n2=$(n2).png")
# end

# Let's slow down, and do a sanity check
ind = get_index(5,m_min)
rho2 = zeros(Complex128, Npoints, Npoints)
rho2[ind,ind] = 1
D = eig(rho2)
ergotropy(rho2,m_min,m_max)

# Let's look at the angular momentum tomography of rho2(t).
pwd_backup = pwd()
cd("V0=10_n1,n2=2_Ltomography/k=8")
t_sample_list = t_sample_list_list[6][1:10:end]
statesample_list = statesample_list_list[6][1:10:end]

for ind = 1:length(t_sample_list)
    println("ploting rho2 heatmaps: $(ind) out of $(length(t_sample_list))...")
    rho2_t = rho2fromCm(statesample_list[ind])
    real_rho2_t = real.(rho2_t)
    imag_rho2_t = imag.(rho2_t)
    heatmap(m_list,m_list,real_rho2_t)
    title!("real(rho2(t)) at t=$(t_sample_list[ind])")
    savefig("$(ind)_t=$(t_sample_list[ind])_real.png")
    heatmap(m_list,m_list,imag_rho2_t)
    title!("imag(rho2(t)) at t=$(t_sample_list[ind])")
    savefig("$(ind)_t=$(t_sample_list[ind])_imag.png")
end

cd(pwd_backup)

####################################################################
# check the off-diagonal terms of rho2_t
saved_dict = load("V0=10_n1=2_n2=2/data_n1=2_n2=2_V0=10.0_k_min=3_k_max=10.jld")
t_sample_list = saved_dict["t_sample_list_list"][6]
statesample_list = saved_dict["statesample_list_list"][6]
saved_dict=0;gc();

t_sample_list = t_sample_list[1:10:end]
statesample_list = statesample_list[1:10:end]
rho2_t = [rho2fromCm(Cm_t) for Cm_t in statesample_list]
for rho2 in rho2_t
    for ind = 1:Npoints
        rho2[ind,ind]=0.0
    end
end

# now plot the off-diagonal heatmaps at each t.
pwd_backup = pwd()
cd("V0=10_n1=2_n2=2")
mkdir("off_diagonal_heatmaps")
cd("off_diagonal_heatmaps")
for ind = 1:length(t_sample_list)
    t_sample = t_sample_list[ind]
    rho_t = real(rho2_t[ind])
    heatmap(m_list,m_list,rho_t)
    title!("off diagonals: real(rho2) at t=$(t_sample)")
    savefig("$(ind)_t=$(t_sample)_real_offdiagonal.png")
end
cd(pwd_backup)
