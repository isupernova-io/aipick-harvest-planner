"""Harvest environment (measured pose in, predicted pose after a pick) — the substrate both the heuristics and the policy run on.

Extracted from aipick_vecenv.ipynb so that the learned policy and the baselines it is measured
against execute the same code. A benchmark whose baseline runs through a different path is not a
benchmark; that lesson was paid for once already in this project.

    import aipick_env as E
    E.load()                       # trees + outcome model
    s = E.run_shift(E.policy_selective, threshold=0.9, seed=0)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ---- geometry and platform ------------------------------------------------------------------
ARM_HALF_X, ARM_LIFT = 0.423, 1.420
ARM_DEPTH_LO, ARM_DEPTH_HI = -0.201, 0.646
R_STAND, ZONE_LO, ZONE_HI, TOUCH = 0.45, 1.2, 4.5, 1.12
NEIGHBOUR_MAX_D, SIZE_EST_ABS = 4.0, 0.003

X_GRID = np.arange(-1.20, 1.201, 0.05)
H_GRID = np.arange(ZONE_LO - ARM_LIFT, ZONE_HI, 0.05)

TREE_SPACING, SHIFT_HOURS = 0.90, 1.0
TRAVEL, LIFT_SPEED, PICK_SECONDS = 0.30, 0.15, 14.0
DETACH_SHARE = 0.449          # measured: share of knocks that pass the abscission limit

STATION_X = {"left": -0.60, "front": 0.0, "right": 0.60}

# filled by load()
MODEL = FEATS = CLASSES = FILLS = UTIL_VEC = None
DYN = None                 # settled-pose classifier; None keeps the measured pose
DYN_THRESH = 0.5
# The aperture the dynamics model was fitted against, copied from generate.aperture_from
# rather than refitted. Importing the generator here would drag the detections into a module
# that has no business loading them, and a fitted approximation would drift from the real
# quantisation. Keep these three in step with generate.py if they ever change.
APERTURE_QUANT = 0.002
APERTURE_MIN_RATIO = 0.85
APERTURE_MAX_RATIO = 1.60


def APERTURE_OF(dim_est, ratio=1.0):
    a = round(ratio*dim_est/APERTURE_QUANT)*APERTURE_QUANT
    return float(np.clip(a, APERTURE_MIN_RATIO*dim_est, APERTURE_MAX_RATIO*dim_est))
SENTINEL = 99.0
T = None
STATIONS = None


def load(root=None, trees="trees_measured_pose.csv", dynamics=True,
         data=None, models=None):
    """Read the canopy, the outcome model and the settled-pose classifier.

    Paths are explicit. Earlier versions assembled them from a single root, which meant
    a tree laid out any other way could hold every artifact and still not load.

        load(ROOT)                                   the usual case
        load(data=DATA, models=MODELS)               anything else
        load(ROOT, dynamics=False)                   measured pose, no update after a pick
    """
    global MODEL, FEATS, CLASSES, FILLS, UTIL_VEC, SENTINEL, T, STATIONS, DYN
    root = Path(root) if root else Path(__file__).resolve().parent.parent
    data = Path(data) if data else root/"data"
    models = Path(models) if models else root/"models"
    with open(models/"outcome.pkl", "rb") as f:
        m = pickle.load(f)
    MODEL, FEATS, CLASSES = m["model"], m["features"], list(m["classes"])
    FILLS, SENTINEL = m["sentinel_fills"], m["sentinel"]
    UTIL_VEC = np.array([m["utility"][c] for c in CLASSES])
    T = pd.read_csv(data/trees)
    STATIONS = [c[4:] for c in T.columns if c.startswith("vis_") and c != "vis_any"]

    # The model was fitted from a DataFrame and warns when handed a bare array. The array is what
    # makes this fast and its column order is correct -- but correct should be checked, because a
    # silent mis-ordering would not show up anywhere in the outputs.
    trained = list(getattr(MODEL, "feature_name_", FEATS))
    assert trained == FEATS, f"feature order differs from training:\n{trained}\n{FEATS}"
    DYN = None
    if dynamics:
        p = models/"dynamics.joblib"
        if p.exists():
            with open(p, "rb") as f:
                DYN = pickle.load(f)

    import warnings
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    return T


def _sweep_basis(ax):
    sweep = np.tile(np.array([1.0, 0.0, 0.0]), (len(ax), 1))
    sweep -= (sweep*ax).sum(1, keepdims=True)*ax
    return sweep/np.maximum(np.linalg.norm(sweep, axis=1, keepdims=True), 1e-9)


class TreeGeometry:
    """Pairwise quantities for one tree. Static: removing a fruit does not change them."""

    def __init__(self, df):
        f = df.reset_index(drop=True)
        self.ids = f.fruit_id.to_numpy()
        self.n = len(f)
        self.P = f[["x", "y", "z"]].to_numpy(float)
        self.dim = f.dim.to_numpy(float)
        ax = f[["sdir_x", "sdir_y", "sdir_z"]].to_numpy(float)
        self.ax = ax/np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-9)
        self.vis = {s: f[f"vis_{s}"].to_numpy(bool) for s in STATIONS}

        rel = self.P[None, :, :] - self.P[:, None, :]
        self.DIST = np.linalg.norm(rel, axis=2)
        np.fill_diagonal(self.DIST, np.inf)
        self.NDIAM = self.DIST/self.dim[:, None]
        self.WITHIN = self.DIST <= NEIGHBOUR_MAX_D*self.dim[:, None]

        along = (rel*self.ax[:, None, :]).sum(2)
        self.ALONG = along/self.dim[:, None]
        perp = rel - along[:, :, None]*self.ax[:, None, :]
        self.RADIAL = np.linalg.norm(perp, axis=2)/self.dim[:, None]

        sweep = _sweep_basis(self.ax)
        cross = np.cross(self.ax, sweep)
        unit = rel/np.maximum(self.DIST[:, :, None], 1e-12)
        self.BCOS = (unit*sweep[:, None, :]).sum(2)
        self.BSIN = (unit*cross[:, None, :]).sum(2)

        ratio = self.DIST/((self.dim[:, None] + self.dim[None, :])/2)
        self.LEAN = np.where(ratio <= TOUCH,
                             np.interp(ratio, [0.95, 1.12, 1.60], [25.0, 19.5, 0.0]), 0.0)

        # The pose the generator measured, precomputed once per fruit. The interpolation
        # table this replaces returned zero or about twenty degrees and correlated 0.140 with
        # what the physics settles to; the outcome model was fitted on measured values.
        if "lean_deg" in f.columns:
            self.LEAN0 = f.lean_deg.to_numpy(float)
            self.SAX = f[["sax_x", "sax_y", "sax_z"]].to_numpy(float)
        else:
            self.LEAN0 = np.zeros(self.n)
            self.SAX = np.column_stack([np.zeros(self.n), np.zeros(self.n), np.ones(self.n)])

        self.dim_est = np.array([
            max(float(d) + np.random.default_rng(int(i)).normal(0, SIZE_EST_ABS), 0.020)
            for i, d in zip(self.ids, self.dim)])
        self.COL = {c: k for k, c in enumerate(FEATS)}

    def observe_all(self, alive, aperture_ratio=1.0, rows=None):
        idx = np.arange(self.n) if rows is None else np.asarray(rows)
        if not len(idx):
            return np.zeros((0, len(FEATS)))
        cand = self.WITHIN[idx] & alive[None, :]
        cand[np.arange(len(idx)), idx] = False
        n_nb = cand.sum(1)
        has = n_nb > 0
        d = np.where(cand, self.DIST[idx], np.inf)
        k = d.argmin(1)

        out = np.zeros((len(idx), len(FEATS)))
        C = self.COL
        bcos = np.where(has, self.BCOS[idx, k], 0.0)
        bsin = np.where(has, self.BSIN[idx, k], 0.0)
        lean = self.LEAN0[idx]
        out[:, C["obs_ax_x"]] = self.SAX[idx, 0]
        out[:, C["obs_ax_y"]] = self.SAX[idx, 1]
        out[:, C["obs_ax_z"]] = self.SAX[idx, 2]
        out[:, C["obs_lean_deg"]] = lean
        out[:, C["obs_nearest_diam_c"]] = np.where(has, self.NDIAM[idx, k],
                                                   FILLS["obs_nearest_diam"])
        out[:, C["obs_has_neighbour"]] = has.astype(float)
        out[:, C["obs_n_neighbours"]] = np.minimum(n_nb, 2).astype(float)
        out[:, C["obs_bearing_cos"]] = bcos
        out[:, C["obs_bearing_sin"]] = bsin
        out[:, C["obs_dim_est"]] = self.dim_est[idx]
        out[:, C["obs_visible_frac"]] = 0.36
        out[:, C["obs_n_points"]] = 16000.0
        out[:, C["aperture_ratio"]] = aperture_ratio
        out[:, C["obs_nb_along_c"]] = np.where(has, self.ALONG[idx, k], FILLS["obs_nb_along"])
        out[:, C["obs_nb_radial_c"]] = np.where(has, self.RADIAL[idx, k], FILLS["obs_nb_radial"])
        return out

    def disturbed_by(self, j):
        """Fruit that must be re-observed when j goes. Beyond four diameters nothing moves."""
        return np.where(self.WITHIN[:, j])[0]


class TreeState:
    """Who is left on one tree, and what the outcome model currently thinks of them."""

    def __init__(self, geom, aperture_ratio=1.0):
        self.g = geom
        self.alive = np.ones(geom.n, bool)
        self.aperture_ratio = aperture_ratio
        self.proba = np.zeros((geom.n, len(CLASSES)))
        self.dirty = np.ones(geom.n, bool)
        self.upright = np.zeros(geom.n, bool)

    def _obs(self, rows=None):
        out = self.g.observe_all(self.alive, self.aperture_ratio, rows=rows)
        idx = np.arange(self.g.n) if rows is None else np.asarray(rows)
        up = self.upright[idx]
        if up.any():
            C = self.g.COL
            out[up, C["obs_lean_deg"]] = 0.0
            out[up, C["obs_ax_x"]] = 0.0
            out[up, C["obs_ax_y"]] = 0.0
            out[up, C["obs_ax_z"]] = 1.0
        return out

    def _dyn_rows(self, nb, j, obs):
        g, C = self.g, self.g.COL
        dim = g.dim[nb]
        n_surv = max(min(int(self.alive[g.WITHIN[:, j]].sum()) - 1, 3), 1)
        f = {"pre_lean_deg": obs[:, C["obs_lean_deg"]],
             "pre_ax_x": obs[:, C["obs_ax_x"]], "pre_ax_y": obs[:, C["obs_ax_y"]],
             "pre_ax_z": obs[:, C["obs_ax_z"]],
             "pre_bearing_cos": obs[:, C["obs_bearing_cos"]],
             "pre_bearing_sin": obs[:, C["obs_bearing_sin"]],
             "pre_along": g.ALONG[nb, j]*dim,        # stored per diameter, fitted in metres
             "pre_radial": g.RADIAL[nb, j]*dim,
             "pre_dist": g.DIST[nb, j],
             "aperture": np.full(len(nb), APERTURE_OF(g.dim_est[j])),
             "dim_mm": dim*1000.0,
             "n_surv": np.full(len(nb), float(n_surv)),
             "n_units": np.full(len(nb), float(n_surv) + 1.0)}
        return np.column_stack([f[c] for c in DYN["features"]])

    def _refresh(self):
        rows = np.where(self.dirty & self.alive)[0]
        if len(rows):
            self.proba[rows] = MODEL.predict_proba(self._obs(rows=rows))
            self.dirty[rows] = False

    def utilities(self):
        self._refresh()
        return np.where(self.alive, self.proba @ UTIL_VEC, -np.inf)

    def features(self):
        self._refresh()
        return self._obs()

    def probabilities(self, j):
        self._refresh()
        return self.proba[j]

    def remove(self, j):
        nb = self.g.disturbed_by(j)
        nb = nb[self.alive[nb] & (nb != j)]
        if DYN is not None and len(nb) and int(self.alive[self.g.WITHIN[:, j]].sum()) - 1 >= 2:
            p = DYN["model"].predict_proba(self._dyn_rows(nb, j, self._obs(rows=nb)))[:, 1]
            self.upright[nb] |= p < DYN_THRESH
        self.alive[j] = False
        self.dirty[self.g.disturbed_by(j)] = True


def build_positions(geom, x_grid=X_GRID, h_grid=H_GRID, half_x=ARM_HALF_X, lift=ARM_LIFT):
    fx, fy, fz = geom.P[:, 0], geom.P[:, 1], geom.P[:, 2]
    depth_ok = (ARM_DEPTH_LO <= R_STAND - fz) & (R_STAND - fz <= ARM_DEPTH_HI)
    pos, rows = [], []
    for x in x_grid:
        station = min(STATION_X, key=lambda s: abs(STATION_X[s] - x))
        base_ok = geom.vis[station] & (np.abs(fx - x) <= half_x) & depth_ok
        if not base_ok.any():
            continue
        for h in h_grid:
            dy = fy - h
            ok = base_ok & (dy >= 0) & (dy <= lift)
            if ok.any():
                pos.append((float(x), float(h)))
                rows.append(ok)
    return np.array(pos), np.array(rows)


def move_seconds(a, b):
    return 0.0 if a is None else abs(b[0]-a[0])/TRAVEL + abs(b[1]-a[1])/LIFT_SPEED


def cheapest_position(POS, COV, j, pos):
    """Least motion among positions that reach fruit j."""
    rows = np.where(COV[:, j])[0]
    if not len(rows):
        return None, np.inf
    if pos is None:
        return tuple(POS[rows[0]]), 0.0
    cost = np.abs(POS[rows, 0]-pos[0])/TRAVEL + np.abs(POS[rows, 1]-pos[1])/LIFT_SPEED
    k = int(cost.argmin())
    return tuple(POS[rows[k]]), float(cost[k])


_CACHE = {}


def tree_cache(tid, half_x=ARM_HALF_X, lift=ARM_LIFT):
    key = (tid, half_x, lift)
    if key not in _CACHE:
        g = TreeGeometry(T[T.tree == tid])
        _CACHE[key] = (g, *build_positions(g, half_x=half_x, lift=lift))
    return _CACHE[key]


def run_shift(policy, cycle=PICK_SECONDS, threshold=0.0, seed=0, tree_ids=range(6),
              budget=SHIFT_HOURS*3600, half_x=ARM_HALF_X, lift=ARM_LIFT, collect=False):
    """One shift down a row.

    policy(state, POS, COV, pos, ctx) -> (fruit index, position) or None to leave this tree.
    With collect=True the per-decision record needed for a policy gradient is returned too.
    """
    rng = np.random.default_rng(seed)
    secs = 0.0
    picked = lost = damaged = attempts = trees = 0
    trace = []

    for tid in tree_ids:
        if secs >= budget:
            break
        geom, POS, COV = tree_cache(tid, half_x, lift)
        st = TreeState(geom)
        pos = None
        while secs < budget:
            ctx = dict(cycle=cycle, threshold=threshold, secs=secs, budget=budget,
                       half_x=half_x, lift=lift, rng=rng)
            out = policy(st, POS, COV, pos, ctx)
            if out is None:
                break
            if collect:
                j, p, rec = out
                trace.append(rec)
            else:
                j, p = out
            secs += move_seconds(pos, p) + cycle
            pos = p
            pr = st.probabilities(j)
            outcome = CLASSES[int(rng.choice(len(CLASSES), p=pr))]
            st.remove(j)
            attempts += 1
            if outcome == "SUCCESS":
                picked += 1
            elif outcome == "DEFECT":
                picked += 1
                damaged += 1
            elif outcome == "NEIGHBOR_KNOCKED":
                picked += 1
                d = np.where(st.alive, geom.DIST[j], np.inf)
                nb = int(d.argmin())
                if np.isfinite(d[nb]) and rng.random() < DETACH_SHARE:
                    st.remove(nb)
                    lost += 1
        trees += 1
        secs += TREE_SPACING/TRAVEL

    hours = min(secs, budget)/3600
    premium = picked - damaged
    out = dict(premium=premium, picked=picked, lost=lost, damaged=damaged,
               attempts=attempts, trees=trees, seconds=min(secs, budget),
               premium_per_hour=premium/hours if hours else 0.0)
    return (out, trace) if collect else out


# ---- baselines --------------------------------------------------------------------------------
def policy_selective(st, POS, COV, pos, ctx):
    """Best expected utility per second, declining anything below the threshold.

    The number a learned policy has to beat.
    """
    u = st.utilities()
    ok = COV.any(axis=0) & st.alive & (u >= ctx["threshold"])
    if not ok.any():
        return None
    best, arg = -np.inf, None
    for j in np.where(ok)[0]:
        p, cost = cheapest_position(POS, COV, j, pos)
        if p is None:
            continue
        rate = u[j]/(cost + ctx["cycle"])
        if rate > best:
            best, arg = rate, (int(j), p)
    return arg


def policy_bottom_up(st, POS, COV, pos, ctx):
    live = COV.any(axis=0) & st.alive
    if not live.any():
        return None
    j = int(np.argmin(np.where(live, st.g.P[:, 1], np.inf)))
    p, _ = cheapest_position(POS, COV, j, pos)
    return None if p is None else (j, p)
