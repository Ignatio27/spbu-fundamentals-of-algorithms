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
    matrix = np.array(A, dtype=float)
    dim = matrix.shape[0]
    orthogonal_matrix = np.zeros_like(matrix)
    upper_triangle = np.zeros((matrix.shape[1], matrix.shape[1]))
    iterations = 50
    for i in range(matrix.shape[1]):
        vec = matrix[:, i]
        for j in range(i):
            upper_triangle[j, i] = orthogonal_matrix[:, j] @ matrix[:, i]
            vec -= upper_triangle[j, i] * orthogonal_matrix[:, j]

        upper_triangle[i, i] = np.linalg.norm(vec)
        orthogonal_matrix[:, i] = vec / upper_triangle[i, i]
    for _ in range(iterations):
        matrix = upper_triangle @ orthogonal_matrix
    eigen_vals = np.diag(matrix)
    return eigen_vals
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
