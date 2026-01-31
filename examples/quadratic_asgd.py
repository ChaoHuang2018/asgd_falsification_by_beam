# project/examples/quadratic_asgd.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from core.system import AsyncSystem
from core.remodeling import QuadraticRemodeling
from core.subspace import make_basis, project_history_rows

@dataclass
class QuadraticObjective:
    """
    f(x) = 1/2 x^T H x, H PSD (enforced by projection)
    """
    H: np.ndarray  # (n,n)

    def __post_init__(self):
        self.H = np.array(self.H, copy=True, dtype=float)
        n = self.H.shape[0]
        if self.H.shape != (n, n):
            raise ValueError("H must be square.")

        # 1) Force symmetry (quadratic form semantics)
        self.H = 0.5 * (self.H + self.H.T)

        # 2) Project to PSD (eigenvalue clipping)
        #    This guarantees soundness for UB/pruning that assumes nonnegative quadratic cost.
        evals, evecs = np.linalg.eigh(self.H)
        evals_clipped = np.maximum(evals, 0.0)
        self.H = (evecs * evals_clipped) @ evecs.T

        # 3) Re-symmetrize for numerical cleanliness
        self.H = 0.5 * (self.H + self.H.T)


    @property
    def n(self) -> int:
        return self.H.shape[0]

    def loss(self, x: np.ndarray) -> float:
        return 0.5 * float(x.T @ self.H @ x)

    def apply_H(self, x: np.ndarray) -> np.ndarray:
        return self.H @ x


@dataclass
class QuadraticASGDSystem(AsyncSystem):
    """
    Standard linear ASGD semantics used in the remodeling:

      x_{t+1} = x_t - eta * (1/K) * sum_k H x_{t-d_k}

    State z is a history stack: z[j] = x_{t-j}, for j=0..D
    z shape: (D+1, n)
    """
    objective: QuadraticObjective
    K: int
    D: int
    eta: float
    init_scale: float = 1.0
    seed: int = 0

    # NEW: initial-set subspace restriction (strict semantics for S-reachable-on-subspace certificates)
    # "none"      : no restriction
    # "mask_last3": for n=4, enforce x4=0 (3D subspace span(e1,e2,e3))
    # "ones_perp3": for n=4, enforce sum_i x_i = 0 (3D subspace 1^⊥)
    init_subspace: str = "none"

    init_subspace_mode: str = "none"   # "none" | "mask" | "ones_perp"
    init_subspace_dim: int = 0         # r; 0 means "no subspace" (or main sets explicitly)

    # If set, init_state() returns this history stack (pointwise experiments)
    _fixed_init_state: Optional[np.ndarray] = field(default=None, init=False, repr=False)


    # 在 QuadraticASGDSystem 类内加入一个小函数（可选，但清晰）：
    def _apply_init_subspace(self, z: np.ndarray) -> np.ndarray:
        mode = str(self.init_subspace_mode).lower()
        r = int(self.init_subspace_dim)

        if mode == "none" or r <= 0:
            return z

        n = int(self.objective.n)
        if r > n:
            raise ValueError(f"init_subspace_dim r={r} cannot exceed n={n}")

        # ones_perp requires r <= n-1
        if mode in ("ones_perp", "ones-perp", "sum0", "sum_zero") and r > n - 1:
            raise ValueError(f"ones_perp subspace requires r<=n-1, got n={n}, r={r}")

        U = make_basis(n=n, r=r, mode=mode)  # (n x r) orthonormal
        return project_history_rows(z, U)

    def _project_init_subspace(self, z: np.ndarray) -> np.ndarray:
        """Project/clip each history vector z[j] into the chosen subspace (if any)."""
        mode = self.init_subspace
        if mode == "none":
            return z

        n = self.objective.n
        if n != 4:
            raise ValueError(f"init_subspace='{mode}' currently implemented for n=4 only, got n={n}")

        if mode == "mask_last3":
            # enforce x4 = 0 for every history vector
            z = np.array(z, copy=True)
            z[:, 3] = 0.0
            return z

        if mode == "ones_perp3":
            # enforce sum_i x_i = 0 for every history vector
            z = np.array(z, copy=True)
            z = z - np.mean(z, axis=1, keepdims=True)
            return z

        raise ValueError(f"Unknown init_subspace mode: {mode}")

    def init_state(self, seed: Optional[int] = None) -> np.ndarray:
        """Return initial history stack z0 of shape (D+1, n).

        - If a fixed init has been set via set_fixed_init_state(), return that (copy).
        - Otherwise sample a Gaussian history stack using (seed or self.seed) and init_scale.
        - In both cases, apply the configured init_subspace projection (if any).
        """
        if self._fixed_init_state is not None:
            z = np.array(self._fixed_init_state, copy=True)
            z = self._apply_init_subspace(z)
            return z

        if seed is None:
            seed = self.seed
        rng = np.random.default_rng(seed)
        n = self.objective.n
        z = rng.normal(size=(self.D + 1, n)) * self.init_scale
        z = self._apply_init_subspace(z)
        return z

    def set_fixed_init_state(self, z0: np.ndarray) -> None:
        """Force init_state() to return the provided history stack (pointwise experiments)."""
        z0 = np.asarray(z0, dtype=float)
        if z0.shape != (self.D + 1, self.objective.n):
            raise ValueError(f"fixed z0 must have shape {(self.D+1, self.objective.n)}, got {z0.shape}")
        self._fixed_init_state = np.array(z0, copy=True)

    def clear_fixed_init_state(self) -> None:
        """Undo set_fixed_init_state()."""
        self._fixed_init_state = None

    def sample_init_state_ball(self, rng: np.random.Generator, radius: float) -> np.ndarray:
        """Sample z0 uniformly from an L2 ball in the lifted history-stack space.

        Sample s ~ Unif({||s||_2 <= radius}) in R^{(D+1)*n} and reshape to (D+1, n),
        then apply init_subspace projection (if configured).
        """
        n = int(self.objective.n)
        dim = int((self.D + 1) * n)
        v = rng.normal(size=(dim,))
        nv = float(np.linalg.norm(v))
        if nv < 1e-15:
            v[0] = 1.0
            nv = 1.0
        v = v / nv
        # uniform-in-ball scaling: r ~ U^{1/dim}
        r = float(radius) * float(rng.random() ** (1.0 / dim))
        s = v * r
        z = s.reshape(self.D + 1, n)
        z = self._apply_init_subspace(z)
        return z

    def describe_initial_set(self, kind: str, radius: Optional[float] = None) -> str:
        """Human-readable initial set description (used by main/experiments)."""
        n = int(self.objective.n)
        dim = int((self.D + 1) * n)
        mode = str(self.init_subspace_mode).lower()
        r = int(self.init_subspace_dim)
        sub = "none" if mode == "none" or r <= 0 else f"{mode}(r={r})"

        if kind == "ball":
            rad = float(radius) if radius is not None else 1.0
            return f"X0 = L2-ball in R^{dim} (history-stack), radius={rad:g}, subspace={sub}"
        if kind == "gaussian":
            return f"X0 = Gaussian N(0, init_scale^2 I) in R^{dim} (history-stack), init_scale={self.init_scale:g}, subspace={sub}"
        return f"X0 = (unknown kind='{kind}') in R^{dim} (history-stack), subspace={sub}"


    def instant_loss(self, z: np.ndarray) -> float:
        x_t = z[0]
        return self.objective.loss(x_t)

    def step(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        # a: delay vector d (K,)
        d = a.astype(int, copy=False)
        x_t = z[0]
        # aggregate stale contributions
        agg = np.zeros_like(x_t)
        for dk in d:
            agg += self.objective.apply_H(z[dk])
        x_next = x_t - self.eta * (agg / self.K)

        # shift history
        z_next = np.empty_like(z)
        z_next[0] = x_next
        z_next[1:] = z[:-1]
        return z_next

    def make_remodeling(self) -> QuadraticRemodeling:
        return QuadraticRemodeling(H=self.objective.H, eta=self.eta, D=self.D, K=self.K)

    def get_Q(self) -> np.ndarray:
        """
        Q for area objective under the lifted history-stack state s_t = [x_t, x_{t-1}, ..., x_{t-D}].
        If instant_loss uses x_t^T H x_t, then
            Q = diag(H, 0, ..., 0).
        """
        H = np.asarray(self.objective.H, dtype=float)
        n = int(H.shape[0])
        D = int(self.D)
        m = (D + 1) * n

        Q = np.zeros((m, m), dtype=float)
        Q[:n, :n] = 0.5 * (H + H.T)
        return Q
    
    @property
    def H(self) -> np.ndarray:
        return self.objective.H

