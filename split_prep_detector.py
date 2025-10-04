"""
split_prep_pipeline_clean.py

A super-thin, human-readable pipeline for testing generalized noncontextuality
in PM (prepare–measure) and PTM (prepare–transform–measure) scenarios.

Pipeline (one screen):
  1) prepare_scenario              -> S0, E, groups, Ts
  2) early_OEs (from S0, E)        -> A_prep (prep OEs), A_meas (measurement OEs/OEM)
     build_vertices                -> Φ (OEM-aware), Ψ (prep-OE-aware)
  3) make_data_ptm                 -> p_ptm
  3) make_data_pm                  -> S_PM, p_pm
  4) discover_identities           -> float OEs + exact/LLL transform identities
  5) (optional) detect_equalities  -> prep+transform==prep (logging only)
  6) solve_pm_model                -> PM feasible?
     solve_ptm_model               -> PTM feasible?
  7) save_csvs + summarize         -> CSVs + console report
"""

from __future__ import annotations
import csv
import numpy as np
from dataclasses import dataclass
from itertools import combinations, permutations
from fractions import Fraction
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linprog

# =============================================================================
# === Outputs & Data containers ===============================================
# =============================================================================

@dataclass
class Outputs:
    """File paths for pipeline outputs."""
    float_csv: str = "identities_float.csv"          # float/SVD-found identities
    nice_csv:  str = "identities_nice.csv"           # integer (LLL) transform identities
    eq_csv:    str = "prep_transform_equalities.csv" # optional logging of coincidences

@dataclass
class Scenario:
    """Fixed objects that define a scenario instance."""
    S0: np.ndarray                  # 4×N0 affine states (homogeneous Bloch coords with leading 1)
    E: np.ndarray                   # 4×K  affine effects (columns are effects)
    groups: Tuple[int, ...]         # measurement block sizes (e.g., (2,2,2) for Z/X/Y)
    Ts: List[np.ndarray]            # list of 4×4 affine transformations
    state_names: List[str]          # labels for S0 columns
    transform_names: List[str]      # labels for Ts

# PTM and PM data are split explicitly for readability
@dataclass
class DataPTM:
    p_ptm: np.ndarray      # (K, N0, T)

@dataclass
class DataPM:
    S_PM: np.ndarray       # (4, N_PM)
    p_pm: np.ndarray       # (K, N_PM)

# Combined container (only used by summarize())
@dataclass
class Data:
    p_ptm: np.ndarray
    S_PM: np.ndarray
    p_pm: np.ndarray

@dataclass
class Identities:
    """Operational identities discovered from the concrete S0/E/Ts."""
    T_float_ok: List[np.ndarray]    # linear transform identities (float/SVD, verified)
    prep_PM: List[np.ndarray]       # prep OE rows on S_PM
    prep_PT: List[np.ndarray]       # prep OE rows on S0
    meas: List[np.ndarray]          # effect OE rows on E  (OEM)
    nice_T: List[np.ndarray]        # integer transform identities (exact + LLL)

@dataclass
class Feasibility:
    pm_ok: bool
    ptm_ok: bool

# =============================================================================
# === Scenario (states, effects, transforms) ===================================
# =============================================================================

def _stabilizer_states_effects() -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    """
    Single-qubit stabilizer scenario in homogeneous Bloch coords:
      States S0: six Pauli eigenstates as columns (1, x, y, z)^T.
      Effects E: ± eigenprojectors as affine half-spaces: e = 0.5*(1, ±n).
    Measurements: Z, X, Y — three binary blocks => groups=(2,2,2).
    """
    bloch = {"Z+":(0,0,1),"Z-":(0,0,-1),"X+":(1,0,0),"X-":(-1,0,0),"Y+":(0,1,0),"Y-":(0,-1,0)}
    S = np.column_stack([[1.0, x, y, z] for (_, (x,y,z)) in bloch.items()])  # 4×6
    def e_pm(nx,ny,nz,sgn): return 0.5*np.array([1.0, sgn*nx, sgn*ny, sgn*nz], float)
    E = np.column_stack([
        e_pm(0,0,1,+1), e_pm(0,0,1,-1),  # Z±
        e_pm(1,0,0,+1), e_pm(1,0,0,-1),  # X±
        e_pm(0,1,0,+1), e_pm(0,1,0,-1),  # Y±
    ])  # 4×6
    return S, E, (2,2,2)

def _state_names() -> List[str]:
    return ["Z+","Z-","X+","X-","Y+","Y-"]

def _Rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)

def _affine_from_R(R: np.ndarray) -> np.ndarray:
    T = np.eye(4); T[1:4,1:4] = R; return T

def _transforms_baldi4() -> List[np.ndarray]:
    Rs = [np.eye(3), _Rz(np.pi), _Rz(+np.pi/2), _Rz(-np.pi/2)]
    return [_affine_from_R(R) for R in Rs]

def _transforms_clifford24() -> List[np.ndarray]:
    Rs = []
    for p in permutations(range(3)):
        P = np.zeros((3,3)); 
        for i,j in enumerate(p): P[i,j] = 1.0
        for sx in (+1,-1):
            for sy in (+1,-1):
                for sz in (+1,-1):
                    Sg = np.diag([sx,sy,sz]); R = Sg @ P
                    if np.linalg.det(R) > 0.5: Rs.append(R.astype(float))
    uniq = []
    for R in Rs:
        if not any(np.allclose(R, Q) for Q in uniq): uniq.append(R)
    assert len(uniq) == 24, f"expected 24, got {len(uniq)}"
    return [_affine_from_R(R) for R in uniq]

def prepare_scenario(scenario: str) -> Scenario:
    """
    Step 1: choose S0/E and the transform list Ts for the named scenario.
    """
    S0, E, groups = _stabilizer_states_effects()
    if scenario == "baldi4":
        Ts = _transforms_baldi4(); tnames = ["I","Z","S","S^-1"]
    elif scenario == "clifford24":
        Ts = _transforms_clifford24(); tnames = [f"T{j}" for j in range(24)]
    else:
        raise ValueError("scenario must be 'baldi4' or 'clifford24'")
    return Scenario(S0=S0, E=E, groups=groups, Ts=Ts,
                    state_names=_state_names(), transform_names=tnames)

# =============================================================================
# === Early OEs (compute BEFORE vertex enumeration) ============================
# =============================================================================

def meas_OEs_from_E(E: np.ndarray, tol: float) -> np.ndarray:
    """
    Measurement operational equivalences (OEM) as right-nullspace rows r with E r ≈ 0.
    Shape: (C_meas, K). May be empty ((0,K)).
    """
    rows = []
    for r in _effect_equalities_svd(E, tol):
        if _is_zero(E @ r, tol):
            rows.append(_nrm_row(r))
    return np.vstack(rows) if rows else np.zeros((0, E.shape[1]))

def prep_OEs_from_S0(S0: np.ndarray, tol: float) -> np.ndarray:
    """
    Preparation OEs on base states: right-nullspace rows r with S0 r ≈ 0.
    Shape: (C_prep, N0). May be empty.
    """
    rows = []
    for r in _state_equalities_svd(S0, tol):
        if _is_zero(S0 @ r, tol):
            rows.append(_nrm_row(r))
    return np.vstack(rows) if rows else np.zeros((0, S0.shape[1]))

# =============================================================================
# === Generic vertex enumeration for {x >= 0 | A_eq x = b_eq} ==================
# =============================================================================

def _enumerate_vertices_nonneg_equalities(Aeq: np.ndarray, beq: np.ndarray,
                                          K: int,
                                          zero_atol: float = 1e-10) -> np.ndarray:
    """
    Enumerate vertices (BFS) of P = { x ∈ R^K | x ≥ 0, Aeq x = beq }.

    Method: choose supports of size rank(Aeq), solve Aeq[:,S] x_S = beq, set others to 0,
    accept if x ≥ -atol and Aeq x == beq (within tol). Deduplicate.
    """
    Aeq = np.atleast_2d(np.asarray(Aeq, float))
    beq = np.atleast_1d(np.asarray(beq, float))
    r = np.linalg.matrix_rank(Aeq)
    verts = []
    seen = set()

    for supp in combinations(range(K), r):
        As = Aeq[:, supp]
        if np.linalg.matrix_rank(As) < r:
            continue
        try:
            xs = np.linalg.lstsq(As, beq, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        x = np.zeros(K)
        x[list(supp)] = xs
        x[np.abs(x) < zero_atol] = 0.0
        if np.all(x >= -zero_atol) and np.allclose(Aeq @ x, beq, atol=1e-8):
            x = np.maximum(x, 0.0)
            key = tuple(np.round(x, 12).tolist())
            if key not in seen:
                seen.add(key)
                verts.append(x)

    return np.array(verts) if verts else np.zeros((0, K))

# =============================================================================
# === Vertices (Φ with OEM; Ψ with prep-OEs) ===================================
# =============================================================================

def build_vertices(groups: Tuple[int, ...],
                   N_states: int,
                   A_meas: Optional[np.ndarray] = None,
                   A_prep: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 2: build the two vertex sets used by the LPs:

      Φ: vertices of { x ≥ 0 | per-measurement sums = 1, A_meas x = 0 }.
         These are the OEM-aware noncontextual measurement assignments
         (indeterministic extremals are included if OEMs require them).

      Ψ: vertices of { ψ ≥ 0 | 1^T ψ = 1, A_prep ψ = 0 }.
         These are the extremal prep assignments compatible with prep OEs.
    """
    # Measurement side
    K = sum(groups)
    Aeq_blocks = np.zeros((len(groups), K))
    beq_blocks = np.ones(len(groups))
    offsets = np.cumsum((0,) + groups[:-1])
    for b, g in enumerate(groups):
        Aeq_blocks[b, offsets[b]:offsets[b]+g] = 1.0
    if A_meas is not None and A_meas.size > 0:
        Aeq = np.vstack([Aeq_blocks, A_meas])
        beq = np.hstack([beq_blocks, np.zeros(A_meas.shape[0])])
    else:
        Aeq = Aeq_blocks
        beq = beq_blocks
    Φ = _enumerate_vertices_nonneg_equalities(Aeq, beq, K)

    # Preparation side
    Aeq_prep = [np.ones(N_states)]
    beq_prep = [1.0]
    if A_prep is not None and A_prep.size > 0:
        Aeq_prep += [row for row in np.atleast_2d(A_prep)]
        beq_prep += [0.0] * int(np.atleast_2d(A_prep).shape[0])
    Aeq_prep = np.array(Aeq_prep, float)
    beq_prep = np.array(beq_prep, float)
    Ψ = _enumerate_vertices_nonneg_equalities(Aeq_prep, beq_prep, N_states)

    return Φ, Ψ

# =============================================================================
# === Data (Born rule) – split PM/PTM lines ===================================
# =============================================================================

def make_data_ptm(S0: np.ndarray, E: np.ndarray, Ts: List[np.ndarray]) -> DataPTM:
    """
    Step 3a: PTM data — p_ptm[k, s, t] = e_k^T ( T_t S0[:, s] ).
    """
    p_ptm = _ptm_data(S0, E, Ts)
    return DataPTM(p_ptm=p_ptm)

def make_data_pm(S0: np.ndarray, E: np.ndarray, Ts: List[np.ndarray], tol: float) -> DataPM:
    """
    Step 3b: PM data — identity-aware closure S_PM and its table p_pm = E^T S_PM.

    S_PM := unique columns in { T S0 : T ∈ Ts }  (always)
            ∪ { S0 } iff an identity transform is present in Ts (within tol).
    """
    S_PM = _closure_states(S0, Ts, tol)
    p_pm  = E.T @ S_PM
    return DataPM(S_PM=S_PM, p_pm=p_pm)

def make_data(S0: np.ndarray, E: np.ndarray, Ts: List[np.ndarray], tol: float) -> Data:
    """Helper: both PTM and PM data (for summarize)."""
    pm  = make_data_pm(S0, E, Ts, tol)
    ptm = make_data_ptm(S0, E, Ts)
    return Data(p_ptm=ptm.p_ptm, S_PM=pm.S_PM, p_pm=pm.p_pm)

def _closure_states(S: np.ndarray, Ts: List[np.ndarray], tol: float) -> np.ndarray:
    """
    Identity-aware closure of states under Ts, with sup-norm deduplication by tol.
    """
    cols = []

    def add(v):
        for u in cols:
            if np.max(np.abs(u - v)) <= tol:
                return
        cols.append(v.copy())

    I_present = any(np.allclose(T, np.eye(4), atol=tol) for T in Ts)

    # Always add T @ S for each T
    for T in Ts:
        W = T @ S
        for j in range(W.shape[1]):
            add(W[:, j])

    # Add raw S columns only if identity is present among Ts
    if I_present:
        for j in range(S.shape[1]):
            add(S[:, j])

    return np.column_stack(cols) if cols else np.zeros((S.shape[0], 0))

def _ptm_data(S: np.ndarray, E: np.ndarray, Ts: List[np.ndarray]) -> np.ndarray:
    _, Ns = S.shape; _, K = E.shape; Tn = len(Ts)
    p = np.zeros((K, Ns, Tn))
    for t, T in enumerate(Ts):
        p[:, :, t] = E.T @ (T @ S)
    return p

# =============================================================================
# === Identities (float/SVD + exact/LLL for transforms) =======================
# =============================================================================

def discover_identities(S0: np.ndarray, S_PM: np.ndarray, E: np.ndarray, Ts: List[np.ndarray],
                        tol: float, max_den: int, lll_delta: float) -> Identities:
    """
    Step 4: extract linear identities that hold in this concrete instance.
    """
    # float/SVD
    T_flat_null = _find_transform_equalities_svd(Ts, tol)
    T_float_ok = [normalize_row(a) for a in T_flat_null if _residual_ok(Ts, a, tol)]
    prep_PM = [_nrm_row(r) for r in _state_equalities_svd(S_PM, tol) if _is_zero(S_PM @ r, tol)]
    prep_PT = [_nrm_row(r) for r in _state_equalities_svd(S0,   tol) if _is_zero(S0   @ r, tol)]
    meas    = [_nrm_row(r) for r in _effect_equalities_svd(E,   tol) if _is_zero(E    @ r, tol)]
    # exact+LLL
    nice_T  = _compute_nice_transform_identities(Ts, max_den, lll_delta)
    return Identities(T_float_ok=T_float_ok, prep_PM=prep_PM, prep_PT=prep_PT, meas=meas, nice_T=nice_T)

def _state_equalities_svd(S, tol):
    U, s, Vt = np.linalg.svd(np.asarray(S), full_matrices=True)
    return Vt[(s > tol).sum():, :]

def _effect_equalities_svd(E, tol):
    U, s, Vt = np.linalg.svd(np.asarray(E), full_matrices=True)
    return Vt[(s > tol).sum():, :]

def _find_transform_equalities_svd(Ts, tol):
    Tflat = np.column_stack([np.asarray(T).reshape(-1, 1) for T in Ts])
    U, s, Vt = np.linalg.svd(Tflat, full_matrices=True)
    return Vt[(s > tol).sum():, :]

def _residual_ok(Ts, alpha, tol):
    R = sum(float(alpha[t]) * Ts[t] for t in range(len(Ts)))
    return float(np.max(np.abs(R))) <= tol

def normalize_row(a, zero_tol: float = 1e-12):
    return _nrm_row(a, zero_tol)

def _nrm_row(a, zero_tol: float = 1e-12):
    a = np.array(a, float)
    a[np.abs(a) < zero_tol] = 0.0
    m = np.max(np.abs(a)) if a.size else 0.0
    if m > 0:
        a = a / m
        for x in a:
            if abs(x) > 0:
                if x < 0: a = -a
                break
    return a

def _is_zero(v, tol): return float(np.max(np.abs(v))) <= tol

# ----- exact + LLL for transform identities ----------------------------------

def _as_fraction_matrix(Ts, max_den):
    Tflat = np.column_stack([np.asarray(T).reshape(-1, 1) for T in Ts])
    m, n = Tflat.shape
    return [[Fraction(Tflat[i, j]).limit_denominator(max_den) for j in range(n)] for i in range(m)]

def _rref_fraction(A):
    A = [row[:] for row in A]
    m = len(A); n = len(A[0]) if m else 0
    i = 0; pivots = []
    for j in range(n):
        pivot = next((r for r in range(i, m) if A[r][j] != 0), None)
        if pivot is None: continue
        A[i], A[pivot] = A[pivot], A[i]
        piv = A[i][j]
        A[i] = [x / piv for x in A[i]]
        for r in range(m):
            if r != i and A[r][j] != 0:
                fac = A[r][j]
                A[r] = [A[r][c] - fac * A[i][c] for c in range(n)]
        pivots.append(j); i += 1
        if i == m: break
    return A, pivots

def _rational_nullspace(A):
    R, pivots = _rref_fraction(A)
    n = len(A[0]) if A else 0
    free = [j for j in range(n) if j not in set(pivots)]
    basis = []
    for f in free:
        v = [Fraction(0) for _ in range(n)]; v[f] = Fraction(1)
        for i_row, j_piv in enumerate(pivots):
            s = Fraction(0)
            for j in range(n):
                if j == j_piv: continue
                if R[i_row][j] != 0 and v[j] != 0:
                    s += R[i_row][j] * v[j]
            v[j_piv] = -s
        basis.append(v)
    return basis

def _gcd(a, b):
    a = abs(int(a)); b = abs(int(b))
    while b: a, b = b, a % b
    return a

def _integer_kernel_basis(basis_frac):
    out = []
    for v in basis_frac:
        dens = [f.denominator for f in v]
        lcm = 1
        for d in dens:
            a, b = lcm, d
            while b: a, b = b, a % b
            lcm = (lcm // a) * d
        ints = [int(f.numerator * (lcm // f.denominator)) for f in v]
        g = 0
        for x in ints: g = abs(x) if g == 0 else _gcd(g, x)
        if g > 0: ints = [x // g for x in ints]
        for x in ints:
            if x != 0:
                if x < 0: ints = [-y for y in ints]
                break
        out.append(np.array(ints, dtype=int))
    return out

def _lll(B: np.ndarray, delta: float = 0.75) -> np.ndarray:
    B = B.astype(np.int64).copy()
    n, k = B.shape
    def gs(Bf):
        U = np.zeros_like(Bf)
        mu = np.zeros((k, k))
        norm2 = np.zeros(k)
        for i in range(k):
            vi = Bf[:, i].copy()
            for j in range(i):
                mu[i, j] = np.dot(vi, U[:, j]) / norm2[j] if norm2[j] != 0 else 0.0
                vi -= mu[i, j] * U[:, j]
            U[:, i] = vi; norm2[i] = np.dot(vi, vi)
        return U, mu, norm2
    def size_reduce(i, j, B, mu):
        q = int(round(mu[i, j]))
        if q != 0:
            B[:, i] -= q * B[:, j]
            mu[i, :j+1] -= q * mu[j, :j+1]
            mu[i, j] -= q
    Bf = B.astype(np.float64)
    U, mu, norm2 = gs(Bf)
    i = 1
    while i < k:
        for j in range(i-1, -1, -1): size_reduce(i, j, B, mu)
        Bf = B.astype(np.float64); U, mu, norm2 = gs(Bf)
        if norm2[i] >= (delta - mu[i, i-1]**2) * norm2[i-1]:
            i += 1
        else:
            B[:, [i, i-1]] = B[:, [i-1, i]]
            i = max(i-1, 1)
    return B

def _compute_nice_transform_identities(Ts: List[np.ndarray], max_den: int, delta: float) -> List[np.ndarray]:
    A = _as_fraction_matrix(Ts, max_den)
    basis_frac = _rational_nullspace(A)
    if not basis_frac: return []
    int_basis = _integer_kernel_basis(basis_frac)
    if not int_basis: return []
    B = np.stack(int_basis, axis=1)
    B_red = _lll(B, delta=delta)
    cand = [B_red[:, i].copy() for i in range(B_red.shape[1])] + [b.copy() for b in int_basis]
    normed, seen = [], set()
    for v in cand:
        g = 0
        for x in v: g = abs(x) if g == 0 else _gcd(g, x)
        if g > 1: v = v // g
        for x in v:
            if x != 0:
                if x < 0: v = -v
                break
        key = tuple(v.tolist())
        if key not in seen and np.any(v != 0):
            seen.add(key); normed.append(v)
    normed.sort(key=lambda vv: (int(np.count_nonzero(vv)), int(np.sum(np.abs(vv))), int(np.max(np.abs(vv)))))
    return normed

# =============================================================================
# === Optional diagnostics: prep+transform==prep ===============================
# =============================================================================

def detect_equalities(S0: np.ndarray, Ts: List[np.ndarray], p_ptm: np.ndarray,
                      tol: float,
                      scenario_name: str,
                      state_names: List[str],
                      tnames: List[str]) -> List[Dict]:
    """
    Step 5 (optional): detect triples (s, t, s2) with T_t S0[:,s] ≈ S0[:,s2] within tol.
    """
    eqs = _detect_prep_transform_equalities(S0, Ts, tol)
    I_idx = _find_identity_index(Ts, tol)
    ver = _verify_equalities_in_ptm(p_ptm, I_idx, eqs)
    rows = []
    for (s, t, s2, state_res), (_, _, _, prob_res) in zip(eqs, ver):
        rows.append({
            "scenario": scenario_name,
            "t_idx": t, "t_name": tnames[t] if 0 <= t < len(tnames) else f"T{t}",
            "s_idx": s, "s_name": state_names[s] if 0 <= s < len(state_names) else f"s{s}",
            "s2_idx": s2, "s2_name": state_names[s2] if 0 <= s2 < len(state_names) else f"s{s2}",
            "state_residual_inf": f"{state_res:.3e}",
            "prob_residual_inf": f"{prob_res:.3e}" if np.isfinite(prob_res) else "NA(no I)",
            "note": "prep+transform==prep (within tol)"
        })
    return rows

def _find_identity_index(Ts, tol):
    I = np.eye(4)
    for i, T in enumerate(Ts):
        if np.allclose(T, I, atol=tol): return i
    return -1

def _detect_prep_transform_equalities(S, Ts, tol):
    S = np.asarray(S); Ns = S.shape[1]; out = []
    for s in range(Ns):
        v = S[:, s]
        for t, T in enumerate(Ts):
            w = T @ v
            diffs = np.max(np.abs(S - w.reshape(-1,1)), axis=0)
            for s2 in np.where(diffs <= tol)[0]:
                res = float(np.max(np.abs(S[:, s2] - w)))
                out.append((s, t, s2, res))
    seen, clean = set(), []
    for s, t, s2, res in out:
        key = (s, t, s2)
        if key not in seen:
            seen.add(key); clean.append((s, t, s2, res))
    return clean

def _verify_equalities_in_ptm(p_ptm, I_idx, equalities):
    K, N, T = p_ptm.shape; out = []
    for (s, t, s2, _) in equalities:
        if 0 <= I_idx < T:
            dev = float(np.max(np.abs(p_ptm[:, s2, I_idx] - p_ptm[:, s, t])))
        else:
            dev = float('nan')
        out.append((s, t, s2, dev))
    return out

# =============================================================================
# === Feasibility LPs – split PM/PTM lines ====================================
# =============================================================================

def solve_pm_model(Φ: np.ndarray, p_pm: np.ndarray, OEP_PM: Optional[np.ndarray]) -> bool:
    """
    Step 6a: PM feasibility LP (no transform constraints here).
    """
    OEP_PM_arr = np.vstack(OEP_PM) if (OEP_PM is not None and len(OEP_PM)>0) else None
    return _pm_feasibility(Φ, p_pm, OEP_PM_arr)

def solve_ptm_model(Φ: np.ndarray, Ψ: np.ndarray, p_ptm: np.ndarray,
                    alphas_T: Optional[np.ndarray],
                    A_prep:   Optional[np.ndarray],
                    A_meas:   Optional[np.ndarray]) -> bool:
    """
    Step 6b: PTM feasibility LP with linear identities enforced.
    """
    return _ptm_feasibility(Φ, Ψ, p_ptm, alphas_T, A_prep, A_meas)

# ---- Core LP builders --------------------------------------------------------

def _pm_feasibility(Phi, p_pm, OEP_eq=None):
    """
    PM LP in standard form: variables x[s,kp] ≥ 0 over measurement vertices Φ[kp,:].
    Equalities:
      (1) ∑_{kp} x[s,kp] = 1                        (per-prep normalization)
      (2) ∑_{s} A[c,s] * x[s,kp] = 0               (prep OEs on closure)
      (3) ∑_{kp} Phi[kp,k] * x[s,kp] = p_pm[k,s]   (match data)
    """
    Phi = np.asarray(Phi); Kp, K = Phi.shape
    p = np.asarray(p_pm); assert p.shape[0] == K
    N = p.shape[1]
    C = 0 if OEP_eq is None else np.asarray(OEP_eq).shape[0]
    def idx(s,kp): return s*Kp + kp
    n = N*Kp; Aeq, beq = [], []
    # (1) normalization per prep
    for s in range(N):
        row = np.zeros(n); row[slice(s*Kp,(s+1)*Kp)] = 1.0
        Aeq.append(row); beq.append(1.0)
    # (2) prep OEs pushed onto Φ-weights
    if C > 0:
        A = np.asarray(OEP_eq, float)
        for c in range(C):
            for kp in range(Kp):
                row = np.zeros(n)
                for s in range(N): row[idx(s,kp)] = A[c, s]
                Aeq.append(row); beq.append(0.0)
    # (3) reproduce probabilities
    for s in range(N):
        for k in range(K):
            row = np.zeros(n)
            for kp in range(Kp): row[idx(s,kp)] = Phi[kp, k]
            Aeq.append(row); beq.append(p[k, s])
    Aeq = np.vstack(Aeq); beq = np.array(beq, float)
    bounds = [(0,None)]*n; c = np.zeros(n)
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    return bool(res.success)

def _build_ptm_eq(Phi, Psi, p_data, alphas_T, A_prep, A_meas):
    """
    Build A_eq x = b_eq for PTM LP with variables x[t,kp,l] ≥ 0.

    Constraints encoded:
      (NORM)    per-transform normalization
      (FLAG)    equal marginals over l across t (flag-convexification)
      (T-OEs)   linear transform identities (Σ_t a_t x[t,·,·] = 0)
      (PREP-OE) push A_prep through Ψ
      (MEAS-OE) push A_meas through Φ
      (DATA)    reproduce p_data
    """
    Phi = np.asarray(Phi); Kp, K = Phi.shape
    Psi = np.asarray(Psi); L, N = Psi.shape
    _, _, T = p_data.shape
    def idx(t,kp,l): return t*Kp*L + kp*L + l
    n = T*Kp*L
    Aeq, beq = [], []

    # (NORM)
    for t in range(T):
        row = np.zeros(n)
        for kp in range(Kp):
            for l in range(L): row[idx(t,kp,l)] = 1.0
        Aeq.append(row); beq.append(1.0)

    # (FLAG)
    if T >= 2:
        t0 = 0
        for l in range(L):
            for t in range(1, T):
                row = np.zeros(n)
                for kp in range(Kp):
                    row[idx(t,kp,l)] += 1.0
                    row[idx(t0,kp,l)] -= 1.0
                Aeq.append(row); beq.append(0.0)

    # (T-OEs)
    if alphas_T is not None and np.size(alphas_T)>0:
        Aalpha = np.atleast_2d(np.asarray(alphas_T, float))
        for c in range(Aalpha.shape[0]):
            a = Aalpha[c]
            for kp in range(Kp):
                for l in range(L):
                    row = np.zeros(n)
                    for t in range(T): row[idx(t,kp,l)] = a[t]
                    Aeq.append(row); beq.append(0.0)

    # (PREP-OE)
    if A_prep is not None and np.size(A_prep)>0:
        A_prep = np.atleast_2d(np.asarray(A_prep, float))
        alpha_map = np.array([A_prep @ Psi[l, :].reshape(-1,1) for l in range(L)]).squeeze(-1)
        for t in range(T):
            for kp in range(Kp):
                for c in range(A_prep.shape[0]):
                    row = np.zeros(n)
                    for l in range(L): row[idx(t,kp,l)] = alpha_map[l, c]
                    Aeq.append(row); beq.append(0.0)

    # (MEAS-OE)
    if A_meas is not None and np.size(A_meas)>0:
        A_meas = np.atleast_2d(np.asarray(A_meas, float))
        beta_map = np.array([A_meas @ Phi[kp, :].reshape(-1,1) for kp in range(Kp)]).squeeze(-1)
        for t in range(T):
            for l in range(L):
                for c in range(A_meas.shape[0]):
                    row = np.zeros(n)
                    for kp in range(Kp): row[idx(t,kp,l)] = beta_map[kp, c]
                    Aeq.append(row); beq.append(0.0)

    # (DATA)
    for t in range(T):
        for s in range(N):
            for k in range(K):
                row = np.zeros(n)
                for kp in range(Kp):
                    for l in range(L):
                        row[idx(t,kp,l)] = Phi[kp, k] * Psi[l, s]
                Aeq.append(row); beq.append(p_data[k, s, t])

    Aeq = np.vstack(Aeq) if Aeq else np.zeros((0, n))
    beq = np.array(beq, float)
    bounds = [(0, None)] * n
    return Aeq, beq, bounds

def _ptm_feasibility(Phi, Psi, p_data, alphas_T, A_prep, A_meas):
    Aeq, beq, bounds = _build_ptm_eq(Phi, Psi, p_data, alphas_T, A_prep, A_meas)
    c = np.zeros(Aeq.shape[1])
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    return bool(res.success)

# =============================================================================
# === CSV + summary ============================================================
# =============================================================================

def save_csvs(out: Outputs, scenario_name: str, tol: float, ids: Identities, eq_rows: List[Dict]) -> None:
    # float identities
    float_rows = []
    for j, a in enumerate(ids.T_float_ok):
        float_rows.append(_rec_float(scenario_name, tol, "transform(float-SVD)", j, a, _vec_str(a)))
    for j, a in enumerate(ids.prep_PM):
        float_rows.append(_rec_float(scenario_name, tol, "prep(PM)", j, a, _vec_str(a)))
    for j, a in enumerate(ids.prep_PT):
        float_rows.append(_rec_float(scenario_name, tol, "prep(PTM)", j, a, _vec_str(a)))
    for j, a in enumerate(ids.meas):
        float_rows.append(_rec_float(scenario_name, tol, "meas", j, a, _vec_str(a)))
    _write_csv(out.float_csv,
               ["kind","scenario","T_count","index","coeffs","coeffs_rational","residual_inf","tol","holds","note"],
               float_rows)

    # nice integer transform identities
    nice_rows = []
    for idx, v in enumerate(ids.nice_T):
        nice_rows.append({
            "kind":"transform(nice-LLL)","scenario":scenario_name,"T_count":None,
            "index":idx,"coeffs":str(v.tolist()),"coeffs_rational":str([int(x) for x in v]),
            "residual_inf":"0","tol":tol,"holds":True,"note":"LLL-reduced"
        })
    _write_csv(out.nice_csv,
               ["kind","scenario","T_count","index","coeffs","coeffs_rational","residual_inf","tol","holds","note"],
               nice_rows)

    # optional equality log
    _write_csv(out.eq_csv,
               ["scenario","t_idx","t_name","s_idx","s_name","s2_idx","s2_name","state_residual_inf","prob_residual_inf","note"],
               eq_rows)

def _rec_float(scenario_name, tol, kind, idx, a, a_str):
    return {
        "kind": kind, "scenario": scenario_name, "T_count": None, "index": idx,
        "coeffs": a_str, "coeffs_rational": a_str, "residual_inf": "0",
        "tol": tol, "holds": True, "note": "auto"
    }

def _vec_str(a): return "[" + ", ".join(f"{float(x):.6g}" for x in np.asarray(a).ravel()) + "]"

def _write_csv(path: str, fields: List[str], rows: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow(r)

def summarize(scenario_name: str, scn: Scenario, data: Data, ids: Identities, feas: Feasibility) -> str:
    lines = []
    lines.append(f"[{scenario_name}] PM states (closure): {data.S_PM.shape[1]} | Effects: {scn.E.shape[1]} | Transforms: {len(scn.Ts)}")
    lines.append(f"Identities — T(float): {len(ids.T_float_ok)}, prep(PM): {len(ids.prep_PM)}, prep(PTM): {len(ids.prep_PT)}, meas: {len(ids.meas)}, nice(T): {len(ids.nice_T)}")
    lines.append(f"Feasibility — PM: {feas.pm_ok} | PTM: {feas.ptm_ok}")
    if feas.pm_ok and not feas.ptm_ok:
        lines.append("SPLIT-PREP PARADOX: PM admits a noncontextual model (on closure-preps), PTM does not.")
    return "\n".join(lines)

# =============================================================================
# === THE PIPELINE (small and readable) =======================================
# =============================================================================

def run_pipeline(scenario: str = "baldi4", tol: float = 1e-12, lll_delta: float = 0.75, max_den: int = 1, log_equalities: bool = True, outputs: Optional[Outputs] = None) -> None:
    # Choose output sinks (CSV file names). Pure plumbing—no math here.
    out = outputs if outputs is not None else Outputs()

    # -------------------------------------------------------------------------
    # 1) SCENARIO: fix the operational primitives and their GPT embeddings.
    #    - S0: base preparation columns (affine 4-vectors).
    #    - E : effect columns (affine 4-vectors).
    #    - groups: per-measurement block sizes (e.g., (2,2,2) for Z/X/Y).
    #    - Ts: list of affine 4×4 transforms to be used on the PTM line and
    #          to generate closure preparations on the PM line if identity is present.
    # -------------------------------------------------------------------------
    scn = prepare_scenario(scenario)

    # -------------------------------------------------------------------------
    # 2) OEs: Extract linear operational identities from S0 and E.
    #
    #    - A_meas: effect-side identities (OEM) as rows r with E r ≈ 0.
    #
    #    - A_prep: base preparation identities as rows r with S0 r ≈ 0.
    # -------------------------------------------------------------------------
    A_meas = meas_OEs_from_E(scn.E, tol)     # effect OEs (OEM)
    A_prep = prep_OEs_from_S0(scn.S0, tol)   # base prep OEs

    # -------------------------------------------------------------------------
    # 2) VERTICES: Build Φ and Ψ by vertex enumeration.
    #
    #    Φ: vertices of { x ≥ 0 | per-block sums = 1, A_meas x = 0 }.
    #       These are the extremal noncontextual measurement assignments.
    #
    #    Ψ: vertices of { ψ ≥ 0 | 1^T ψ = 1, A_prep ψ = 0 }.
    #       These are the extremal preparation assignments consistent with OEP.
    #
    #    We pass A_meas and A_prep so the polytopes we enumerate
    #    already encode the declared linear identities.
    # -------------------------------------------------------------------------
    Φ, Ψ = build_vertices(scn.groups, scn.S0.shape[1],
                          A_meas=A_meas_early,
                          A_prep=A_prep_early)

    # -------------------------------------------------------------------------
    # 3) DATA:
    #
    #    PTM: compute p_ptm[k, s, t] = e_k^T ( T_t S0[:, s] ). 
    #    This is the raw prepare–transform–measure table the PTM LP will attempt to fit.
    #
    #    PM : build S_PM by applying every T in Ts to S0; Then compute
    #         p_pm = E^T S_PM. PM has no transform constraints—transforms are
    #         encapsulated as “just more preparations.”
    # -------------------------------------------------------------------------
    ptm = make_data_ptm(scn.S0, scn.E, scn.Ts)      # PTM line -> p_ptm
    pm  = make_data_pm (scn.S0, scn.E, scn.Ts, tol) # PM  line -> S_PM, p_pm

    # -------------------------------------------------------------------------
    # 4) IDENTITIES: Discover linear identities that hold in this concrete
    #    instance using linear algebra:
    #      - SVD nullspaces for states/effects (prep + meas OEs),
    #      - SVD nullspace for transforms (float relations) + exact integer
    #        kernel via rationalization + LLL to produce short integer identities.
    #
    #    We compute prep PM-identities on S_PM for the PM LP,
    #    and prep PT-identities on the base S0 for the PTM LP.
    # -------------------------------------------------------------------------
    ids  = discover_identities(scn.S0, pm.S_PM, scn.E, scn.Ts, tol, max_den, lll_delta)

    # -------------------------------------------------------------------------
    # 5) FEASIBILITY:
    #
    #    PM: We test whether p_pm admits a noncontextual model with respect to
    #        OEM and preparation OEs on S_PM.
    #        No transform constraints exist on the PM line by design.
    #
    #        OEP_PM stacks the prep-OE rows on S_PM; if none, we pass None.
    # -------------------------------------------------------------------------
    OEP_PM = np.vstack(ids.prep_PM) if ids.prep_PM else None
    pm_ok  = solve_pm_model(Φ, pm.p_pm, OEP_PM)

    # -------------------------------------------------------------------------
    #    PTM: We enforce transform identities (alphas_T) together with
    #         prep and meas identities pushed through Ψ and Φ, respectively,
    #         and we require the model to reproduce p_ptm. This is the linearized
    #         PTM feasibility program (flag-convexification viewpoint).
    # -------------------------------------------------------------------------
    alphas_T = np.vstack(ids.T_float_ok) if ids.T_float_ok else None
    A_prep   = np.vstack(ids.prep_PT)    if ids.prep_PT    else None
    A_meas   = np.vstack(ids.meas)       if ids.meas       else None
    ptm_ok   = solve_ptm_model(Φ, Ψ, ptm.p_ptm, alphas_T, A_prep, A_meas)

    # -------------------------------------------------------------------------
    # 7)SUMMARY:
    #    We persist identities and equality logs to CSVs for inspection and
    #    print a human-readable summary.
    # -------------------------------------------------------------------------
    save_csvs(out, scenario, tol, ids, eq_rows)
    data_for_summary = Data(p_ptm=ptm.p_ptm, S_PM=pm.S_PM, p_pm=pm.p_pm)
    print(summarize(scenario, scn, data_for_summary, ids, Feasibility(pm_ok=pm_ok, ptm_ok=ptm_ok)))

# =============================================================================
# === Main ====================================================================
# =============================================================================

if __name__ == "__main__":
    # Example runs (two scenarios).
    run_pipeline(scenario="baldi4",  log_equalities=True,
                 outputs=Outputs("float_baldi4.csv", "nice_baldi4.csv", "eq_baldi4.csv"))
    print("-"*72)
    run_pipeline(scenario="clifford24", log_equalities=True,
                 outputs=Outputs("float_cliff24.csv", "nice_cliff24.csv", "eq_cliff24.csv"))
