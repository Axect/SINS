using DifferentialEquations
using DataFrames, NCDataFrame
using Distributions

size = parse(Int64, ARGS[1])

function H_velo!(du, v, u, p, t)
    dx, dy = v
    du[1] = dx
    du[2] = dy
end

function H_accel!(dv, v, u, p, t)
    x, y = u
    r = (x^2 + y^2)^(3/2)
    dv[1] = -x/r
    dv[2] = -y/r
end

tspan = (0.0, 10.0)

r_dist = Uniform(1.0, 2.0)
th_dist = Uniform(0.0, 2.0*pi)
#e_dist = Normal(0.0, pi/12)
vx_dist = Uniform(-0.2, -0.1)
vy_dist = Uniform(0.1, 0.2)

r = rand(r_dist, size)
th = rand(th_dist, size)
#e = rand(e_dist, size)
x = r .* cos.(th)
y = r .* sin.(th)
vx = rand(vx_dist, size)
vy = rand(vy_dist, size)
#vx = -0.1 * r .* cos.(th .+ e)
#vy = 0.1 * r .* sin.(th .+ e)

z = hcat(x, y, vx, vy)

for i = 1:size
    a, b, va, vb = z[i,:]
    if abs(a/b - va/vb) < 0.1
        va = -va
    end
    prob = DynamicalODEProblem(
        H_accel!,
        H_velo!,
        [va, vb],
        [a, b],
        (0.0, 10.0)
    )
    sol = solve(prob, DPRKN6(), saveat=0.01) # Sympletic integrator

    sol_u = zeros(length(sol), 4)
    for i in 1:length(sol)
        va, vb, a, b = sol.u[i]
        sol_u[i,1] = a
        sol_u[i,2] = b
        sol_u[i,3] = va
        sol_u[i,4] = vb
    end

    df = DataFrame(
        "t" => sol.t,
        "x" => sol_u[:,1],
        "y" => sol_u[:,2],
        "vx" => sol_u[:,3],
        "vy" => sol_u[:,4]
    )
    writenc(df, "data2/kepler2d_$i.nc")
end
