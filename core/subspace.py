# core/subspace.py
from __future__ import annotations
import numpy as np


def basis_mask(n: int, r: int) -> np.ndarray:
    """
    Coordinate-mask subspace: span(e1,...,er), i.e. x_{r+1:n} = 0.
    Returns U in R^{n x r} with orthonormal columns.
    """
    if not (1 <= r <= n):
        raise ValueError(f"basis_mask: require 1<=r<=n, got n={n}, r={r}")
    return np.eye(n, dtype=float)[:, :r]


def basis_ones_perp(n: int, r: int) -> np.ndarray:
    """
    A canonical orthonormal basis for a subspace inside 1^⊥ (sum x_i = 0).

    We first build a deterministic spanning set for 1^⊥:
        v_i = e_i - e_{i+1}, i=1..n-1  (n x (n-1))
    QR => orthonormal Q (n x (n-1)).
    Take first r columns => U (n x r), r <= n-1.

    NOTE: for r < n-1, this picks a fixed r-dim subspace within 1^⊥.
    """
    if n < 2:
        raise ValueError("basis_ones_perp: require n>=2")
    if not (1 <= r <= n - 1):
        raise ValueError(f"basis_ones_perp: require 1<=r<=n-1, got n={n}, r={r}")

    V = np.zeros((n, n - 1), dtype=float)
    for i in range(n - 1):
        V[i, i] = 1.0
        V[i + 1, i] = -1.0

    Q, _ = np.linalg.qr(V)  # deterministic QR for deterministic V
    return Q[:, :r]


def make_basis(n: int, r: int, mode: str) -> np.ndarray:
    mode = str(mode).lower()
    if mode in ("mask", "coord", "coordinate"):
        return basis_mask(n, r)
    if mode in ("ones_perp", "ones-perp", "sum0", "sum_zero"):
        return basis_ones_perp(n, r)
    raise ValueError(f"Unknown subspace mode: {mode}")


def lift_basis(U: np.ndarray, D: int) -> np.ndarray:
    """
    Lift U (n x r) to U_lift ( (D+1)n x (D+1)r ) for history-stack vec state:
        s = vec([x_t, x_{t-1}, ..., x_{t-D}])
    """
    U = np.asarray(U, dtype=float)
    n, r = U.shape
    I = np.eye(D + 1, dtype=float)
    return np.kron(I, U)


def projector(U: np.ndarray) -> np.ndarray:
    """Orthogonal projector P = U U^T (assumes U has orthonormal columns)."""
    U = np.asarray(U, dtype=float)
    return U @ U.T


def project_history_rows(z: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    z shape: (D+1, n) where each row is a vector x_{t-j}^T.
    Project each row onto span(U):
        z_proj = z @ (U U^T)
    """
    z = np.asarray(z, dtype=float)
    P = projector(U)
    return z @ P
