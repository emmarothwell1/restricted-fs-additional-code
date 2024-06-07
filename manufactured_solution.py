"""
This is to solve the problem, with a manufactured solution, in the comparison
to function space section.
"""
from firedrake import *
import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor, tricontour
import pygments
plt.rcParams.update({'font.size': 26})

mesh = UnitSquareMesh(16, 16)
x, y = SpatialCoordinate(mesh)

# solving in the unrestricted case
V = FunctionSpace(mesh, "CG", 2)
u, v = TrialFunction(V), TestFunction(V)
f = Function(V).interpolate(-2 * (y**3 - 1.5 * y**2) + (x-x**2) * (6*y -3))
G = inner(f, v) * dx
a = inner(grad(u), grad(v)) * dx
bc_left = DirichletBC(V, 0.0, 1)
bc_right = DirichletBC(V, 0.0, 2)

u = Function(V)
solve(a == G, u, bcs=[bc_left, bc_right], restrict=False) 

V_res = RestrictedFunctionSpace(V, boundary_set=[1, 2])
u_res, v_res = TrialFunction(V_res), TestFunction(V_res)
f_res = Function(V_res).interpolate(-2 * (y**3 - 1.5 * y**2) + (x-x**2) * (6*y -3))
G_res = inner(f_res, v_res) * dx
a_res = inner(grad(u_res), grad(v_res)) * dx

bc_left_res = DirichletBC(V_res, 0.0, 1)
bc_right_res = DirichletBC(V_res, 0.0, 2)

u_res = Function(V_res)
solve(a_res == G_res, u_res, bcs=[bc_left_res, bc_right_res], restrict=False)

# interpolating the known solution into the function space. 
u_solution = Function(V).interpolate(-1 * x * (1-x) * (y**3 - 1.5 * y**2))

# printing norms of interest. 
print("Error norm between restricted and unrestricted solutions:", errornorm(u, u_res)) 
print("Error norm between unrestricted and true solutions:", errornorm(u, u_solution))
print("Error norm between restricted and true solutions:", errornorm(u_res, u_solution))

# plotting
fig, [ax1, ax2, ax3] = plt.subplots(nrows=1, ncols=3)
plot_u = tripcolor(u, axes=ax1)
plot_solution = tripcolor(u_solution, axes=ax2)
plot_ures = tripcolor(u_res, axes=ax3)

ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.yaxis.set_label_coords(-0.05, 0.5)
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.yaxis.set_label_coords(-0.05, 0.5)
ax1.set_title("Numerical Solution \n(Unrestricted)", wrap=True)
ax2.set_title("Interpolation of True \nSolution", wrap=True)
ax3.set_title("Numerical Solution \n(Restricted)", wrap=True)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(plot_ures, cax=cbar_ax)
plt.show()