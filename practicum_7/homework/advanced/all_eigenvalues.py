from collections import defaultdict
from dataclasses import dataclass
import os
import yaml
import time


import numpy as np
import scipy.io
import scipy.linalg

from collections import namedtuple

import numpy as np
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


@dataclass
class Performance:
    time: float = 0.0
    relative_error: float = 0.0

def get_all_eigenvalues(A: NDArrayFloat) -> NDArrayFloat:
    A = np.copy(A)
    n = A.shape[0]
    Q = np.zeros_like(A)
    R = np.zeros((A.shape[1], A.shape[1]))
    n_iters = 50

    for i in range(A.shape[1]):
        v = A[:, i]

        for j in range(i):
            R[j, i] = Q[:, j] @ A[:, i]
            v = v - R[j, i] * Q[:, j]

        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]

    for _ in range(n_iters):
        A = R @ Q
    eigenvalues = np.diag(A)

    return eigenvalues

def run_test_cases(
    path_to_homework: str, path_to_matrices: str
) -> dict[str, Performance]:
    matrix_filenames = []
    performance_by_matrix = defaultdict(Performance)
    with open(os.path.join(path_to_homework, "matrices.yaml"), "r") as f:
        matrix_filenames = yaml.safe_load(f)
    for i, matrix_filename in enumerate(matrix_filenames):
        print(f"Processing matrix {i+1} out of {len(matrix_filenames)}")
        A = scipy.io.mmread(os.path.join(path_to_matrices, matrix_filename)).todense().A
        perf = performance_by_matrix[matrix_filename]
        t1 = time.time()
        eigvals = get_all_eigenvalues(A)
        t2 = time.time()
        perf.time += t2 - t1
        eigvals1 = eigvals.copy()
        eigvals1.sort()
        eigvals.sort()
        perf.relative_error = np.median(eigvals_exact - eigvals1) / np.abs(eigvals_exact)
    return performance_by_matrix


if __name__ == "__main__":
    path_to_homework = os.path.join("practicum_7", "homework", "advanced")
    path_to_matrices = os.path.join("practicum_6", "homework", "advanced", "matrices")
    performance_by_matrix = run_test_cases(
        path_to_homework=path_to_homework,
        path_to_matrices=path_to_matrices,
    )

    print("\nResult summary:")
    for filename, perf in performance_by_matrix.items():
        print(
            f"Matrix: {filename}. "
            f"Average time: {perf.time:.2e} seconds. "
            f"Relative error: {perf.relative_error:.2e}"
        )
