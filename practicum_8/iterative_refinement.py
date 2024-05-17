import os
from typing import Optional

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from collections import namedtuple

import numpy as np
from numpy._typing import DTypeLike
from numpy.typing import NDArray


ProblemCase = namedtuple("ProblemCase", "input, output")
NDArrayInt = NDArray[np.int_]
NDArrayFloat = NDArray[np.float_]
import numpy as np
import scipy.linalg


def get_scipy_solution(A, b):
    lu_and_piv = scipy.linalg.lu_factor(A)
    return scipy.linalg.lu_solve(lu_and_piv, b)


def get_numpy_eigenvalues(A):
    return np.linalg.eigvals(A)

def relative_error(x_true, x_approx):
    return np.linalg.norm(x_true - x_approx, axis=1) / np.linalg.norm(x_true)

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



def iterative_refinement(
    A: NDArrayFloat, b: NDArrayFloat, solver, n_iters: int, n_ir_iters: int
) -> NDArrayFloat:
    print(f"IR #1 out of {n_ir_iters}")
    ir_solution_history = np.zeros((n_ir_iters,A.shape[0]),dtype = dtype)
    solution_history = conjugate_gradient_method(A,b,n_iters = n_iters,dtype =dtype)
    x_approx = solution_history[-1,:]
    ir_solution_history[0] = x_approx
    for i in range(n_ir_iters):
        print(f"IR #{i+1} out of {n_ir_iters}")
        solution_history = conjugate_gradient_method(A,b-A@x_approx,n_iters = n_iters,dtype=dtype)
        y_approx = solution_history[-1:]
        x_approx = x_approx+y_approx
        ir_solution_history[i] =x_approx
    return ir_solution_history


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

    # Download s1rmq4m1.mtx.gz here:
    # https://math.nist.gov/MatrixMarket/data/misc/cylshell/s1rmq4m1.html
    path_to_matrix = os.path.join(
        "/Users", "ignat", "Desktop", "pershin_homework", "spbu-fundamentals-of-algorithms", "practicum_6", "homework",
        "advanced", "matrices", "s1rmq4m1.mtx.gz"
    )
    A = scipy.io.mmread(path_to_matrix).todense().A
    b = np.ones((A.shape[0],))
    exact_solution = get_scipy_solution(A, b)
    n_iters = 1000
    n_ir_iters = 5

    dtype = np.float32
    A = A.astype(np.float32)
    b = b.astype(np.float32)

    # Convergence speed for the conjugate gradient method
    ir_solution_history = iterative_refinement(
        A, b, solver=conjugate_gradient_method, n_iters=n_iters, n_ir_iters=n_ir_iters
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    add_convergence_graph_to_axis(ax, exact_solution, ir_solution_history)
    plt.show()
