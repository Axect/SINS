using DifferentialEquations
using DataFrames, NCDataFrame
using YAActL
using Distributions

size = parse(Int64, ARGS[1])

function kepler2d!(du, u, _p, _t)
    du[1] = u[3]
    du[2] = u[4]
    r = sqrt(u[1]^2 + u[2]^2)
    du[3] = -u[1]/r^3
    du[4] = -u[2]/r^3
end

tspan = (0.0, 10.0)

r_dist = Uniform(1.0, 2.0)
th_dist = Uniform(0.0, 2.0*pi)
vx_dist = Uniform(-0.2, -0.1)
vy_dist = Uniform(0.1, 0.2)

r = rand(r_dist, size)
th = rand(th_dist, size)
x = r .* cos.(th)
y = r .* sin.(th)
vx = rand(vx_dist, size)
vy = rand(vy_dist, size)

z = hcat(x, y, vx, vy)

for i = 1:size
    u_init = z[i,:]
    prob = ODEProblem(kepler2d!, u_init, tspan)
    sol = solve(prob, saveat=0.1)
    sol_u = vcat(sol.u'...)
    df = DataFrame(
        "t" => sol.t,
        "x" => sol_u[:,1],
        "y" => sol_u[:,2],
        "vx" => sol_u[:,3],
        "vy" => sol_u[:,4]
    )
    writenc(df, "data/kepler2d_$i.nc")
end