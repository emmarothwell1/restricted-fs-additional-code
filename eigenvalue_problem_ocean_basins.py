import numpy as np 
import matplotlib.pyplot as plt

"""
This is the code that solves the ocean basins eigenproblem. This is done with
multiple shifts.
"""

plt.rcParams.update({'font.size': 30})

from firedrake import *
Lx, Ly = 1, 1
n0 = 50
mesh = RectangleMesh(n0, n0, Lx, Ly, reorder=None)
V = FunctionSpace(mesh, "CG", 1)
bc = DirichletBC(V, 0, "on_boundary")
n = 300
phi, psi = TestFunction(V), TrialFunction(V)

opts = {"eps_gen_non_hermitian": None, "eps_largest_imaginary": None,
        "st_type": "shift", "eps_target": None,
        "st_pc_factor_shift_type": "NONZERO"} # options for solver

eigenproblem_2_shift = LinearEigenproblem(
        A=phi*psi.dx(0)*dx,
        M=-inner(grad(psi), grad(phi))*dx - psi*phi*dx,
        bcs=bc, bc_shift=2, restrict=False)
eigensolver_2_shift = LinearEigensolver(eigenproblem_2_shift, n_evals=n,
                                        solver_parameters=opts) 
nconv_2_shift = eigensolver_2_shift.solve()


eigenproblem_0_shift = LinearEigenproblem(
        A=phi*psi.dx(0)*dx,
        M=-inner(grad(psi), grad(phi))*dx - psi*phi*dx,
        bcs=bc, bc_shift=0.0, restrict=False)

eigensolver_0_shift = LinearEigensolver(eigenproblem_0_shift, n_evals=n,
                                        solver_parameters=opts) 
nconv_0_shift = eigensolver_0_shift.solve()

print("Number of eigenvalues converged (shift=2):", nconv_2_shift)
print("Number of eigenvalues converged (shift=0):", nconv_0_shift)
print("Number of eigenvalues searched for:", n)

min_n_conv = min(nconv_2_shift, nconv_0_shift)
evals_2_shift_real, evals_2_shift_imag = np.zeros(min_n_conv, dtype=complex), np.zeros(min_n_conv, dtype=complex)
evals_0_shift_real, evals_0_shift_imag = np.zeros(min_n_conv, dtype=complex), np.zeros(min_n_conv, dtype=complex)
for i in range(min_n_conv):
    eval = eigensolver_2_shift.eigenvalue(i)
    evals_2_shift_real[i] = np.real(eval)
    evals_2_shift_imag[i] = np.imag(eval)
    eval2 = eigensolver_0_shift.eigenvalue(i)
    evals_0_shift_real[i] = np.real(eval2)
    evals_0_shift_imag[i] = np.imag(eval2)

plt.scatter(evals_2_shift_real, evals_2_shift_imag, color="b", label=r"Shift=2")
plt.scatter(evals_0_shift_real, evals_0_shift_imag, color="r", label=r"Shift=0")
plt.xscale("log")
plt.title(rf"First {min_n_conv} Eigenvalues [Sorted By Largest Imaginary Part], Created Using Different Shift Values", wrap=True)
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.legend()
plt.show()


eigenproblem_res = LinearEigenproblem(
        A=phi*psi.dx(0)*dx,
        M=-inner(grad(psi), grad(phi))*dx - psi*phi*dx,
        bcs=bc, bc_shift=2, restrict=True)
eigensolver_res = LinearEigensolver(eigenproblem_res, n_evals=n, solver_parameters=opts)
nconv_res = eigensolver_res.solve()

print("Number of eigenvalues converged (restricted):", nconv_res)

evals_real_res = np.zeros(nconv_res, dtype=complex)
evals_imag_res = np.zeros(nconv_res, dtype=complex)
for i in range(nconv_res):
    eval = eigensolver_res.eigenvalue(i)
    evals_real_res[i] = np.real(eval)
    evals_imag_res[i] = np.imag(eval)

overall_min_nconv = min(min_n_conv, nconv_res)

# plotting just up to the min of all three converged values?
plt.scatter(evals_2_shift_real[:overall_min_nconv], evals_2_shift_imag[:overall_min_nconv], color="b", label=r"Shift=2")
plt.scatter(evals_0_shift_real[:overall_min_nconv], evals_0_shift_imag[:overall_min_nconv], color="r", label=r"Shift=0")
plt.scatter(evals_real_res[:overall_min_nconv], evals_imag_res[:overall_min_nconv], color="g", label=r"Restricted")
plt.xscale("log")
plt.title(rf"First {overall_min_nconv} Eigenvalues for the Problem, in the Restricted and Unrestricted Eigensolver", wrap=True)
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.legend()
plt.show()