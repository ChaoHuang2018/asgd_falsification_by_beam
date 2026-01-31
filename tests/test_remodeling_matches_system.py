# project/tests/test_remodeling_matches_system.py
import unittest
import numpy as np

from examples.quadratic_asgd import QuadraticObjective, QuadraticASGDSystem
from core.remodeling import QuadraticRemodeling


def make_spd_H(n: int, condition: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.linspace(1.0, condition, n)
    H = Q @ np.diag(eigs) @ Q.T
    return 0.5 * (H + H.T)


class TestRemodelingMatchesSystem(unittest.TestCase):
    def assert_allclose(self, a, b, atol=1e-10, rtol=1e-10, msg=""):
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=msg)

    def test_single_step_matches_A(self):
        """
        For random z (history stack) and random delay vectors d,
        verify: vec(step(z,d)) == A(d) @ vec(z)
        where vec(z) = [x_t; x_{t-1}; ...; x_{t-D}] in C-order flatten.
        """
        n = 6
        K = 5
        D = 4
        eta = 0.03
        seed = 123

        H = make_spd_H(n=n, condition=20.0, seed=seed)
        obj = QuadraticObjective(H=H)
        system = QuadraticASGDSystem(objective=obj, K=K, D=D, eta=eta, seed=seed, init_scale=1.0)
        remodeling = system.make_remodeling()

        rng = np.random.default_rng(999)
        z = system.init_state(seed=seed)  # (D+1, n)

        # Try multiple random actions
        for _ in range(50):
            d = rng.integers(0, D + 1, size=K, dtype=int)  # delays in [0,D]
            z_next = system.step(z, d)

            A = remodeling.get_A(d)  # ((D+1)n, (D+1)n)
            s = z.reshape(-1)        # ((D+1)n,)
            s_next = z_next.reshape(-1)

            s_next_lin = A @ s
            self.assert_allclose(
                s_next_lin, s_next,
                msg=f"Mismatch for d={d.tolist()}"
            )

    def test_multi_step_matches_repeated_linear_dynamics(self):
        """
        Fix a random action d and verify step-by-step agreement for multiple steps.
        """
        n = 5
        K = 4
        D = 3
        eta = 0.05
        seed = 7

        H = make_spd_H(n=n, condition=10.0, seed=seed)
        obj = QuadraticObjective(H=H)
        system = QuadraticASGDSystem(objective=obj, K=K, D=D, eta=eta, seed=seed, init_scale=1.0)
        remodeling = QuadraticRemodeling(H=H, eta=eta, D=D, K=K)

        rng = np.random.default_rng(2025)
        z = system.init_state(seed=seed)

        d = rng.integers(0, D + 1, size=K, dtype=int)
        A = remodeling.get_A(d)

        # Step-by-step: z_{t+1} = step(z_t,d) and s_{t+1} = A s_t
        s = z.reshape(-1)
        for t in range(30):
            z = system.step(z, d)
            s = A @ s
            self.assert_allclose(
                s, z.reshape(-1),
                msg=f"Mismatch at t={t} for fixed d={d.tolist()}"
            )

    def test_A_depends_only_on_counts(self):
        """
        Verify A(d) depends only on the counts m[j]=#{k: d_k=j}:
        permutations of d yield identical A.
        """
        n = 4
        K = 6
        D = 3
        eta = 0.02
        seed = 11

        H = make_spd_H(n=n, condition=5.0, seed=seed)
        remodeling = QuadraticRemodeling(H=H, eta=eta, D=D, K=K)

        d = np.array([0, 0, 1, 2, 2, 3], dtype=int)
        A1 = remodeling.get_A(d)

        rng = np.random.default_rng(0)
        for _ in range(20):
            perm = rng.permutation(K)
            d2 = d[perm]
            A2 = remodeling.get_A(d2)
            self.assert_allclose(A1, A2, msg=f"A differs for permutation {d2.tolist()}")

    def test_cache_hits(self):
        """
        Basic sanity: repeated calls with same counts should hit the cache and return identical matrices.
        """
        n = 3
        K = 5
        D = 4
        eta = 0.01
        seed = 42

        H = make_spd_H(n=n, condition=3.0, seed=seed)
        remodeling = QuadraticRemodeling(H=H, eta=eta, D=D, K=K)

        d = np.array([0, 1, 1, 3, 4], dtype=int)
        A1 = remodeling.get_A(d)
        A2 = remodeling.get_A(d.copy())
        # They should be numerically identical; object identity may also match but we don't rely on it.
        self.assert_allclose(A1, A2)


if __name__ == "__main__":
    unittest.main()
