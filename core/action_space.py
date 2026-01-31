# project/core/action_space.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class ActionSpace:
    """
    Action = delay vector d of length K, each entry in [0, D], with sum(d) <= B.
    This class supports:
      - enumerate all actions (small regime)
      - enumerate a fixed-sum layer (small regime)
      - balanced allocation for a given sum
      - majorization neighbor moves (Robin Hood)
      - sampling actions (large regime fallback)
    """
    K: int
    D: int
    B: int

    def is_valid(self, d: np.ndarray) -> bool:
        if d.shape != (self.K,):
            return False
        if np.any(d < 0) or np.any(d > self.D):
            return False
        if int(np.sum(d)) > self.B:
            return False
        return True

    def balanced(self, s: int) -> np.ndarray:
        """
        Return a balanced delay vector with sum = s (as evenly distributed as possible),
        then sorted nondecreasing for canonical form.
        Requires 0 <= s <= K*D.
        """
        if s < 0 or s > self.K * self.D:
            raise ValueError(f"balanced(s): s={s} out of range [0, K*D].")
        base = s // self.K
        r = s % self.K
        if base > self.D:
            raise ValueError(f"balanced(s): base={base} exceeds D={self.D}.")
        d = np.full(self.K, base, dtype=int)
        # distribute the remainder by +1 to r entries
        if r > 0:
            d[-r:] += 1
        if np.any(d > self.D):
            raise ValueError("balanced(s): distribution exceeds D, adjust B/K/D.")
        d.sort()
        return d

    def counts(self, d: np.ndarray) -> np.ndarray:
        """
        Convert d -> counts m[j] = number of workers with delay=j.
        """
        if d.shape != (self.K,):
            raise ValueError("counts: invalid shape.")
        m = np.zeros(self.D + 1, dtype=int)
        for dj in d:
            m[int(dj)] += 1
        return m
    
    # --- ActionSpace histogram enumeration (paste inside class ActionSpace) ---
    def enumerate_histograms(self):
        """
        Enumerate histogram actions m = (m_0,...,m_D) where:
        - m_j >= 0 integers
        - sum_j m_j = K
        - sum_j (j * m_j) <= B

        This is an exact enumeration of action equivalence classes for ASGD:
        any two delay-vectors d that share the same histogram m yield the same A(m).
        So using histograms does NOT restrict the adversary.
        """
        K = int(self.K)
        D = int(self.D)
        B = int(self.B)

        out = []

        def rec(j: int, remaining_k: int, remaining_budget: int, m: np.ndarray):
            if j == D:
                # last bin gets the remainder
                m[D] = remaining_k
                if D * remaining_k <= remaining_budget:
                    out.append(m.copy())
                return

            max_mj = remaining_k
            # budget constraint: j * mj <= remaining_budget
            if j > 0:
                max_mj = min(max_mj, remaining_budget // j)

            for mj in range(max_mj + 1):
                m[j] = mj
                rec(j + 1, remaining_k - mj, remaining_budget - j * mj, m)
            m[j] = 0

        rec(0, K, B, np.zeros(D + 1, dtype=int))
        return out


    def histogram_to_delay(self, m):
        """
        Convert a histogram m (length D+1) into a canonical delay vector d (length K)
        by expanding in nondecreasing order: [0]*m0 + [1]*m1 + ... + [D]*mD.
        """
        m = np.asarray(m, dtype=int).reshape(-1)
        if m.shape[0] != int(self.D) + 1:
            raise ValueError("histogram_to_delay expects length D+1.")
        if int(np.sum(m)) != int(self.K):
            raise ValueError("histogram_to_delay expects sum(m)=K.")
        d = []
        for j, cnt in enumerate(m.tolist()):
            d.extend([j] * int(cnt))
        return np.asarray(d, dtype=int)


    def canonicalize_action(self, a):
        """
        Return a canonical representative for an action:
        - if a has length K, treat as delay vector and return histogram
        - if a has length D+1, treat as histogram and return itself
        """
        a = np.asarray(a, dtype=int).reshape(-1)
        if a.shape[0] == int(self.K):
            # delay vector -> histogram
            m = np.zeros(int(self.D) + 1, dtype=int)
            for dk in a.tolist():
                m[int(dk)] += 1
            return m
        if a.shape[0] == int(self.D) + 1:
            return a
        raise ValueError("canonicalize_action: unsupported action length.")


    # -------------------------
    # Enumeration (small regimes)
    # -------------------------
    def enumerate_actions(self) -> Iterable[np.ndarray]:
        """
        Enumerate all d with sum(d) <= B. Exponential in K, use only for small K,D.
        """
        cur = np.zeros(self.K, dtype=int)

        def rec(i: int, remaining_sum: int):
            if i == self.K:
                yield cur.copy()
                return
            # di in [0, D] but must allow enough budget for remaining positions
            for val in range(0, self.D + 1):
                if val > remaining_sum:
                    break
                cur[i] = val
                yield from rec(i + 1, remaining_sum - val)

        yield from rec(0, self.B)

    def enumerate_layer_sorted(self, s: int) -> List[np.ndarray]:
        """
        Enumerate canonical (sorted) delay vectors with sum(d)=s.
        This removes permutation duplicates.
        """
        if s < 0 or s > min(self.B, self.K * self.D):
            return []
        results: List[np.ndarray] = []
        cur = np.zeros(self.K, dtype=int)

        def rec(i: int, last: int, remaining: int):
            if i == self.K:
                if remaining == 0:
                    results.append(cur.copy())
                return
            # enforce nondecreasing order: cur[i] >= last
            for val in range(last, self.D + 1):
                if val > remaining:
                    break
                cur[i] = val
                rec(i + 1, val, remaining - val)

        rec(0, 0, s)
        return results

    # -------------------------
    # Majorization neighbor moves
    # -------------------------
    def more_uniform_neighbors(self, d_sorted: np.ndarray) -> List[np.ndarray]:
        """
        Generate neighbors that are "more uniform" via Robin Hood transfer:
        pick i with max, j with min where max-min >= 2, do (max-1, min+1).
        Input expected sorted nondecreasing. Output sorted.
        """
        if d_sorted.shape != (self.K,):
            raise ValueError("more_uniform_neighbors: invalid shape.")
        d = d_sorted.copy()
        d.sort()

        mn = int(d[0])
        mx = int(d[-1])
        if mx - mn < 2:
            return []

        # positions of min and max
        min_idxs = np.where(d == mn)[0]
        max_idxs = np.where(d == mx)[0]

        neighbors = []
        for i in max_idxs:
            for j in min_idxs:
                # transfer 1 from i (max) to j (min)
                dd = d.copy()
                dd[i] -= 1
                dd[j] += 1
                if np.any(dd < 0) or np.any(dd > self.D):
                    continue
                if int(np.sum(dd)) > self.B:
                    continue
                dd.sort()
                neighbors.append(dd)

        # de-duplicate
        uniq = []
        seen = set()
        for x in neighbors:
            key = tuple(int(v) for v in x)
            if key not in seen:
                seen.add(key)
                uniq.append(x)
        return uniq

    # -------------------------
    # Sampling (large regimes)
    # -------------------------
    def sample_action(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample a valid d under sum<=B by sampling per-entry then rejecting.
        This is simplistic but fine for a baseline.
        """
        for _ in range(10_000):
            d = rng.integers(0, self.D + 1, size=self.K, dtype=int)
            if int(np.sum(d)) <= self.B:
                return d
        # fallback: force balanced(B) if rejection is hard
        return self.balanced(min(self.B, self.K * self.D))

    def candidate_actions(self, max_enum: int, rng: np.random.Generator, n_sample: int) -> List[np.ndarray]:
        """
        Provide a candidate set for search expansions:
          - if total actions <= max_enum: enumerate
          - else: sample n_sample
        """
        # rough upper bound (still huge often); we use a quick heuristic
        # if (D+1)^K is small, enumerate; else sample
        if (self.D + 1) ** self.K <= max_enum:
            return [d for d in self.enumerate_actions()]
        return [self.sample_action(rng) for _ in range(n_sample)]
    
    def enumerate_histogram_vertices(self, relax: bool = True) -> List[np.ndarray]:
        """
        Enumerate vertices of the *continuous* histogram polytope:

            M = { m >= 0, sum_j m_j = K, sum_j (j*m_j) <= B }.

        Why this exists:
        - For pointwise certificates, we often need an upper bound on max_{m in feasible} F(m)
            where F(m) is convex in m (typical when state depends affinely on m and we bound using a quadratic/2-norm).
        - The discrete/integer feasible set is a subset of M, hence:
                max_{integer m} F(m) <= max_{m in M} F(m)
            and for convex F, the maximum over M is attained at a vertex.
        - So scanning vertices gives a SAFE (sufficient) upper bound, typically with only O(D^2) candidates.

        relax=True (recommended for certificates):
        - allow fractional vertices (safe upper bound; may be conservative).
        relax=False:
        - keep only integer vertices (NOT a safe upper bound in general; use only if you have a separate justification).

        Returns:
        List of m vectors of length (D+1), dtype float (relax=True) or int (relax=False).
        """
        K = int(self.K)
        D = int(self.D)
        B = int(self.B)

        verts: List[np.ndarray] = []

        # 1-sparse vertices (budget inactive): m_p = K, others 0, requiring p*K <= B
        for p in range(D + 1):
            if p * K <= B:
                m = np.zeros(D + 1, dtype=float if relax else int)
                m[p] = float(K) if relax else K
                verts.append(m)

        # 2-sparse vertices (budget active): only two bins p<q are nonzero
        # Solve:
        #   m_p + m_q = K
        #   p*m_p + q*m_q = B
        # => m_p = (q*K - B)/(q - p), m_q = K - m_p
        for p in range(D + 1):
            for q in range(p + 1, D + 1):
                denom = q - p
                mp = (q * K - B) / denom
                mq = K - mp
                if mp < -1e-12 or mq < -1e-12:
                    continue

                if relax:
                    m = np.zeros(D + 1, dtype=float)
                    m[p] = float(mp)
                    m[q] = float(mq)
                    verts.append(m)
                else:
                    # integer-only vertices (not generally safe as an upper bound)
                    if abs(mp - round(mp)) > 1e-12 or abs(mq - round(mq)) > 1e-12:
                        continue
                    mp_i = int(round(mp))
                    mq_i = int(round(mq))
                    if mp_i < 0 or mq_i < 0 or mp_i + mq_i != K:
                        continue
                    m = np.zeros(D + 1, dtype=int)
                    m[p] = mp_i
                    m[q] = mq_i
                    verts.append(m)

        # de-duplicate (tolerant for float vertices)
        uniq = {}
        for m in verts:
            key = tuple(np.round(np.asarray(m, dtype=float), 12).tolist())
            uniq[key] = m
        return list(uniq.values())
