# ruff: noqa: N802, N803, E741
"""Geometry utilities for vector operations in FSM-based reaction path methods."""

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from scipy.spatial.distance import euclidean


def distance(v1: NDArray[np.floating], v2: NDArray[np.floating]) -> float:
    """Return the Euclidean distance between two vectors v1 and v2."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(euclidean(v1.flatten(), v2.flatten()))


def magnitude(v: NDArray[np.floating]) -> float:
    """Return the magnitude (L2 norm) of a vector with floor for stability."""
    return float(np.maximum(1e-12, np.sqrt(v.dot(v))))


def normalize(v: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return the normalized version of a vector."""
    return v / magnitude(v)


def calculate_arc_length(string: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute cumulative arc length along a string of molecular geometries."""
    nnodes = string.shape[0]
    L = np.zeros((nnodes,))
    s = np.zeros((nnodes,))
    for i in range(1, nnodes):
        L[i] = magnitude((string[i] - string[i - 1]).flatten())
        s[i] = s[i - 1] + L[i]

    return s


def project_trans_rot(
    a: NDArray[np.floating], b: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Minimizes distance between structures a and b by minimizing rotation and translation."""
    centroid_a = np.mean(a, axis=0, keepdims=True)
    centroid_b = np.mean(b, axis=0, keepdims=True)
    A = a - centroid_a
    B = b - centroid_b
    H = B.T @ A
    U, _S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = centroid_b @ R - centroid_a
    return a.flatten(), (b @ R - t).flatten()


def project_trans_rot_fixed(
    a: NDArray[np.floating], b: NDArray[np.floating], fixed: NDArray[int],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Minimizes distance between structures a and b by minimizing rotation and translation."""
    # Calculate rotation matrix based on fixed atoms
    a_fixed = a[fixed]
    b_fixed = b[fixed]  
    centroid_a = np.mean(a_fixed, axis=0, keepdims=True)
    centroid_b = np.mean(b_fixed, axis=0, keepdims=True)
    A = a_fixed - centroid_a
    B = b_fixed - centroid_b
    H = B.T @ A
    U, _S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = centroid_b @ R - centroid_a
    return a.flatten(), (b @ R - t).flatten()


def generate_inertia_I(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute the moment of inertia tensor for a set of 3D coordinates X."""
    I = np.zeros((3, 3))
    I[0, 0] = np.sum(X[:, 1] ** 2 + X[:, 2] ** 2)
    I[1, 1] = np.sum(X[:, 0] ** 2 + X[:, 2] ** 2)
    I[2, 2] = np.sum(X[:, 0] ** 2 + X[:, 1] ** 2)
    I[0, 1] = -np.sum(X[:, 0] * X[:, 1])
    I[0, 2] = -np.sum(X[:, 0] * X[:, 2])
    I[1, 2] = -np.sum(X[:, 1] * X[:, 2])
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]
    return I


def generate_project_rt(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """Construct a projection operator that removes rigid translations and rotations from X."""
    N = X.shape[0]
    I = generate_inertia_I(X)
    _evals, evecs = scipy.linalg.eigh(I)
    evecs = evecs.T

    # use the convention that the element with largest abs. value
    # within a given evec should be positive
    for i in range(3):
        if np.max(evecs[i]) < np.abs(np.min(evecs[i])):
            evecs[i] *= -1

    P = X @ evecs

    R_rot1 = np.zeros(N * 3)
    R_rot2 = np.zeros(N * 3)
    R_rot3 = np.zeros(N * 3)
    R_x = np.zeros(N * 3)
    R_y = np.zeros(N * 3)
    R_z = np.zeros(N * 3)

    for i in range(N):
        for j in range(3):
            R_rot1[3 * i + j] = P[i, 1] * evecs[j, 2] - P[i, 2] * evecs[j, 1]
            R_rot2[3 * i + j] = P[i, 2] * evecs[j, 0] - P[i, 0] * evecs[j, 2]
            R_rot3[3 * i + j] = P[i, 0] * evecs[j, 1] - P[i, 1] * evecs[j, 0]

    for i in range(N):
        R_x[3 * i] = 1.0
        R_y[3 * i + 1] = 1.0
        R_z[3 * i + 2] = 1.0

    R_x = normalize(R_x)
    R_y = normalize(R_y)
    R_z = normalize(R_z)
    R_rot1 = normalize(R_rot1)
    R_rot2 = normalize(R_rot2)
    R_rot3 = normalize(R_rot3)

    proj = np.eye(N * 3)
    proj -= np.outer(R_x, R_x)
    proj -= np.outer(R_y, R_y)
    proj -= np.outer(R_z, R_z)
    proj -= np.outer(R_rot1, R_rot1)
    proj -= np.outer(R_rot2, R_rot2)
    proj -= np.outer(R_rot3, R_rot3)
    return proj


def generate_project_rt_tan(structure: NDArray[np.floating], tangent: NDArray[np.floating]) -> NDArray[np.floating]:
    """Construct a projection operator orthogonal to translations, rotations, and the tangent vector."""
    proj = generate_project_rt(structure)
    proj_tangent = proj @ tangent
    proj_tangent = normalize(proj_tangent)
    proj -= np.outer(proj_tangent, proj_tangent)
    return proj
