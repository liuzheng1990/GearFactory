include("gear_init.jl")
using PyPlot

Cm0_squared = abs2.(Cm0)
p1 = sum(Cm0_squared,2); p2 = sum(Cm0_squred,1);
plot(m_list,p1,"*",label="Gear 1")
plot(m_list,p2,"",label="Gear 2")
grid()
xlim(-8,9)
