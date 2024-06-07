"""
This is basically a clone of the eigensolver test in Firedrake, with images
added.
"""
from firedrake import * 
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 22})

n = 10
degree = 4

mesh = IntervalMesh(n, 0, pi) # setting the mesh between 0 and pi
V = FunctionSpace(mesh, "CG", degree)

u = TrialFunction(V)
v = TestFunction(V)
a = (inner(grad(u), grad(v))) * dx

bc = DirichletBC(V, 0.0, "on_boundary") # this is equivalent to a bc on subdomain 1 and 2

# the default M is the one in this problem, so no need to specify in construction
eigenprob_res = LinearEigenproblem(a, bcs=bc, bc_shift=1, restrict=True)
eigenprob = LinearEigenproblem(a, bcs=bc, bc_shift=70, restrict=False)

n_evals = 10

eigensolver_res = LinearEigensolver(
    eigenprob_res, n_evals, solver_parameters={"eps_largest_real": None}
)
nconv_res = eigensolver_res.solve()

eigensolver = LinearEigensolver(
    eigenprob, n_evals, solver_parameters={"eps_largest_real": None}
)
nconv = eigensolver.solve()

print("Number of eigenvalues found (restricted):", nconv_res)
print("Number of eigenvalues found (unrestricted):", nconv)

estimates_res = np.zeros(nconv_res, dtype=complex)
estimates = np.zeros(nconv, dtype=complex)
for k in range(max(nconv_res, nconv)):
    if k < nconv_res:
        estimates_res[k] = eigensolver_res.eigenvalue(k)
    if k < nconv:
        estimates[k] = eigensolver.eigenvalue(k)

eigenmode_real, eigenmode_imag = eigensolver.eigenfunction(0)
eigenmode_r_res, eigenmode_i_res = eigensolver_res.eigenfunction(0)

true_values = [x**2 for x in range(1, nconv_res+1)] # known eigenvalues

for eval in true_values[:n_evals]:
    plt.vlines(eval, ymin=-0.001, ymax=0.001, linestyle="--", color="black", zorder=-1)
plt.scatter(np.real(estimates_res)[:n_evals], 0.001 * np.ones(nconv_res)[:n_evals], 
            s=200, color="b", marker="x", label="Restricted", zorder=1)
plt.scatter(np.real(estimates)[:n_evals], -0.001 * np.ones(nconv)[:n_evals], 
            s=200, color="r", marker="s", label="Unrestricted", zorder=1)
plt.scatter(np.real(true_values)[:n_evals], np.zeros(len(true_values))[:n_evals], 
            s=200, color="g", marker="*", label="Exact", zorder=1)

plt.title(f"First {n_evals} Converged Eigenvalues for the 1D Poisson Eigenproblem")
plt.tick_params(axis='y',  which='both', left=False,  right=False, labelleft=False) 
plt.xlabel(r"$\lambda$ (real)")
plt.legend(loc=(0.79, 0.2))
plt.show()
