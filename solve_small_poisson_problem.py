"""
This code is to demonstrate the use of Firedrake in solving a small Poisson 
equation
"""

from firedrake import *
mesh = UnitSquareMesh(1, 1)
x, y = SpatialCoordinate(mesh)
# create Lagrange element of degree 4
V = FunctionSpace(mesh, "Lagrange", 4)

u = TrialFunction(V)
v = TestFunction(V)

# 1 indicates the left side of mesh, 2 for right side
bc_left = DirichletBC(V, 0.0, 1)
bc_right = DirichletBC(V, 0.0, 2)

# create bilinear form, dx means integral
a = inner(grad(u), grad(v)) * dx

# create linear form
f = Function(V).interpolate(x**2)
G = inner(f, v) * dx 

# solve, u_sol holds solution
u_sol = Function(V)
solve(a == G, u_sol, bcs=[bc_left, bc_right])