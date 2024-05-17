from typing import Optional
import os
from numpy.typing import DTypeLike
import scipy.io
import matplotlib.pyplot as plt

import numpy as np
from numpy.typing import NDArray
NDArrayInt = NDArray[np.int_]
NDArrayFloat = NDArray[np.float_]
import scipy.linalg
def get_scipy_solution(A, b):
    lu_and_piv = scipy.linalg.lu_factor(A)
    return scipy.linalg.lu_solve(lu_and_piv, b)


def get_numpy_eigenvalues(A):
    return np.linalg.eigvals(A)


# from src.common import NDArrayFloat
# from src.linalg import get_scipy_solution


def conjugate_gradient_method(
    A: NDArrayFloat,
    b: NDArrayFloat,
    n_iters: Optional[int] = None,
    dtype: Optional[DTypeLike] = None,
) -> NDArrayFloat:
    solution_history = np.zeros((n_iters,A.shape[0]),dtype=dtype)
    x_kk = np.zeros_like(b,dtype=dtype)
    r_kk = b - A@ x_kk
    v_kk = r_kk
    for k in range(n_iters):
        r_kk_norm = r_kk @ r_kk
        t_kk = r_kk_norm / (v_kk @(A@v_kk))
        x_kk = x_kk +t_kk*v_kk
        solution_history[k] = x_kk

        r_kk = r_kk - t_kk * A @ v_kk
        s_kk = (r_kk@r_kk)/ r_kk_norm
        v_kk = r_kk + s_kk*v_kk
    return solution_history

def preconditioned_conjugate_gradient_method(
    A: NDArrayFloat,
    b: NDArrayFloat,
    C_inv: NDArrayFloat,
    n_iters: Optional[int] = None,
    dtype: Optional[DTypeLike] = None,
) -> NDArrayFloat:
    solution_history = np.zeros((n_iters, A.shape[0]), dtype=dtype)
    x_kk = np.zeros_like(b, dtype=dtype)
    r_kk = b - A @ x_kk
    w_kk = C_inv @ r_kk
    v_kk = C_inv.T @ w_kk
    for k in range(n_iters):
        w_kk_norm = w_kk @ w_kk
        t_kk = w_kk_norm / (v_kk @ (A @ v_kk))
        x_kk = x_kk + t_kk * v_kk
        solution_history[k] = x_kk

        r_kk = r_kk - t_kk * A @ v_kk
        w_kk = C_inv @ r_kk
        s_kk = (w_kk @ w_kk) / w_kk_norm
        v_kk = C_inv @ w_kk + s_kk * v_kk
    return solution_history



def relative_error(x_true, x_approx):
    return np.linalg.norm(x_true - x_approx, axis=1) / np.linalg.norm(x_true)


def add_convergence_graph_to_axis(
    ax, exact_solution: NDArrayFloat, solution_history: NDArrayFloat
) -> None:
    n_iters = solution_history.shape[0]
    ax.semilogy(
        range(n_iters),
        relative_error(x_true=exact_solution, x_approx=solution_history),
        "o--",
    )
    ax.grid()
    ax.legend(fontsize=12)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(r"$||x - \tilde{x}|| / ||x||$", fontsize=12)


if __name__ == "__main__":
    np.random.seed(42)

    # Try the following matrices
    # nos5.mtx.gz (pos.def., K = O(10^4))
    # bcsstk14.mtx.gz (pos.def., K = O(10^10))

    path_to_matrix = os.path.join(
        "/Users", "ignat", "Desktop", "pershin_homework", "spbu-fundamentals-of-algorithms", "practicum_6", "homework",
        "advanced", "matrices", "bcsstk14.mtx.gz"
    )
    A = scipy.io.mmread(path_to_matrix).todense().A

    b = np.ones((A.shape[0],))
    exact_solution = get_scipy_solution(A, b)
    n_iters = 1000

    # Convergence speed for the conjugate gradient method
    solution_history = conjugate_gradient_method(A, b, n_iters=n_iters)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    add_convergence_graph_to_axis(ax, exact_solution, solution_history)
    plt.show()

    # Convergence speed for the preconditioned conjugate gradient method

    ##########################
    ### PUT YOUR CODE HERE ###
    ##########################

    C_inv = np.diag(1.0/np.sqrt(np.diag(A)))

    solution_history = preconditioned_conjugate_gradient_method(
        A, b, C_inv, n_iters=n_iters
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    add_convergence_graph_to_axis(ax, exact_solution, solution_history)
    plt.show()

    print()
