"""
Code with comments/blank lines removed
"""

from firedrake import *
mesh = UnitSquareMesh(1, 1)
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", 2)
u = TrialFunction(V)
v = TestFunction(V)
bc_left = DirichletBC(V, 0.0, 1)
bc_right = DirichletBC(V, 0.0, 2)
bcs=[bc_left, bc_right]
a = inner(grad(u), grad(v)) * dx
f = Function(V).interpolate(x**2)
G = inner(f, v) * dx 
K = assemble(a, bcs=bcs)
u_sol = Function(V)
solve(a == G, u_sol, bcs=bcs)

print(K.M.values)
print("u values:", u_sol.dat.data)