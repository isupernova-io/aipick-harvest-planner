"""The planning layer: detections in, a harvest plan out.

Everything above the outcome model lives here — where the base stops, which fruit to attempt in
what order, and what to leave. It was written into three notebooks before it was written into a
module, and the notebooks drifted: one still decoded three stations, one had no sweep stage, and
the figures they produced could not be compared. Re-implementing this logic has cost this project
more than any other single mistake, so there is now one copy.

    import environment as E, planner as PL
    E.load(ROOT, trees="trees_measured_pose.csv", dynamics=True)
    PL.load(ROOT)                                  # the station planner and the pick policy

    case = PL.tree_case(23)
    plan = PL.plan_tree(case)                      # a DataFrame, one row per pick
    print(plan.attrs["summary"])

The default configuration is the one the report describes: twenty stops from the trained planner,
a sweep stage for whatever those twenty miss, no selection threshold, and the rate rule choosing
fruit. Each of those is a keyword argument, and `03_evaluation` and `07_station_sweep` vary them
to produce the comparisons they report.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import environment as E  # noqa: E402

# --- the configuration the report describes --------------------------------------------------
HALF_X, LIFT = 0.200, 0.600     # compact arm: reach either side, mast window
K_STATIONS = 20                 # stops the planner decodes before the sweep
THRESHOLD = 0.0                 # take everything the stops reach
SWEEP = True                    # go back for fruit the first pass could not fit

PLANNER = None
POLICY = None
FMU = FSD = None


# --- networks --------------------------------------------------------------------------------
class StationPlanner(nn.Module):
    """Scores every candidate base position, given what is already covered."""

    def __init__(self, d=96, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Linear(7, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4*d, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.pos_head = nn.Sequential(
            nn.Linear(2*d + 4, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, Xf, COV, pos_xy, arm):
        h = self.encoder(self.embed(Xf).unsqueeze(0)).squeeze(0)
        M = torch.as_tensor(COV, dtype=torch.float32)
        z = torch.cat([(M @ h)/M.sum(1, keepdim=True).clamp(min=1.0), (M @ h)/20.0,
                       pos_xy, arm.unsqueeze(0).expand(len(pos_xy), -1)], dim=1)
        return self.pos_head(z).squeeze(-1)


class PickPolicy(nn.Module):
    """Scores every live fruit plus a stop action."""

    def __init__(self, n_feat=None, d=128, heads=4, layers=3, n_ctx=6):
        super().__init__()
        n_feat = n_feat or (len(E.FEATS) + len(E.CLASSES) + 2)
        self.embed = nn.Linear(n_feat, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4*d, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.ctx = nn.Linear(d + n_ctx, d)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.stop = nn.Parameter(torch.zeros(d))
        self.scale = 1.0/np.sqrt(d); self.clip = 10.0

    def scores(self, x, mask, ctx_vec):
        h = self.encoder(self.embed(x).unsqueeze(0)).squeeze(0)
        m = torch.tensor(np.asarray(mask, dtype=bool))
        graph = h[m].mean(0) if m.any() else h.mean(0)
        q = self.q(self.ctx(torch.cat([graph, ctx_vec])))
        att = self.clip*torch.tanh((self.k(h) @ q)*self.scale)
        stop = self.clip*torch.tanh((self.k(self.stop) @ q)*self.scale)
        return torch.cat([att.masked_fill(~m, -1e9), stop.reshape(1)])


def load(root=None, planner="station_planner_k20.pt", policy="pick_policy.pt"):
    """Load the two networks and the scaling the policy was trained with.

    The scaling is recomputed the way training computed it — trees 0-19, the default arm, the
    unpruned arrays — rather than from whatever the caller happens to have loaded. Handing the
    policy inputs scaled differently from training collapses it to an immediate stop, which
    looks exactly like a weak model and has been diagnosed as one three times. Newer checkpoints
    carry their own statistics; those are used when present.
    """
    global PLANNER, POLICY, FMU, FSD
    root = Path(root or os.environ.get("AIPICK_ROOT", r"C:\aipick\final"))
    models = root/"models"

    PLANNER = StationPlanner()
    PLANNER.load_state_dict(torch.load(models/planner))
    PLANNER.eval()

    ck = torch.load(models/policy)
    POLICY = PickPolicy()
    if isinstance(ck, dict) and "state" in ck:
        POLICY.load_state_dict(ck["state"])
        FMU, FSD = np.asarray(ck["fmu"]), np.asarray(ck["fsd"])
        src = "checkpoint"
    else:
        POLICY.load_state_dict(ck)
        FMU, FSD = _policy_stats()
        src = "recomputed"
    POLICY.eval()

    names = list(E.FEATS) + [f"p_{c}" for c in E.CLASSES] + ["move_secs", "rate"]
    k = names.index("move_secs")
    if abs(FMU[k] - 385.324) > 8:
        raise RuntimeError(
            f"move_secs scaling is {FMU[k]:.1f}, training recorded 385.3. The policy will stop "
            "on the first decision; do not read any number produced downstream.")
    return dict(planner=planner, policy=policy, scaling=src,
                move_secs=(float(FMU[k]), float(FSD[k])))


def _policy_stats(tids=range(20)):
    X = [policy_features(E.TreeState(g), P, C, (0.0, 2.0))
         for g, P, C in (E.tree_cache(t) for t in tids)]
    X = np.concatenate(X)
    return X.mean(0), X.std(0) + 1e-6


# --- one tree --------------------------------------------------------------------------------
def prune_positions(POS, COV):
    """Drop candidate positions whose coverage another position already contains."""
    n = COV.sum(axis=1); keep = []
    for i in np.argsort(-n):
        if n[i] and not any(COV[j, COV[i]].all() for j in keep):
            keep.append(i)
    keep = np.array(sorted(keep))
    return POS[keep], COV[keep]


def tree_case(tid, half_x=HALF_X, lift=LIFT):
    geom, POSf, COVf = E.tree_cache(tid, half_x, lift)
    Pp, Cp = prune_positions(POSf, COVf)
    return dict(tid=tid, geom=geom, POS=Pp, COV=Cp, arm=(half_x, lift),
                util=np.clip(E.TreeState(geom).utilities(), 0, None),
                robot_reach=int(COVf.any(axis=0).sum()))


# --- stations --------------------------------------------------------------------------------
def planner_features(case, covered=None):
    g = case["geom"]; Pc = g.P.copy(); Pc[:, 0] -= Pc[:, 0].mean()
    cov = np.zeros(len(Pc)) if covered is None else covered.astype(float)
    return np.column_stack([Pc, g.dim, case["util"],
                            case["COV"].any(axis=0).astype(float), cov]).astype(np.float32)


def decode_stations(case, k=K_STATIONS):
    """The trained planner, choosing one stop at a time and marking what each covers.

    It has no stopping rule and spends the count it was trained for. On this canopy that beats
    coverage search, which halts at its first zero-gain step around 95% of the reach; the
    planner's last few stops carry it to 98%.
    """
    half_x, lift = case["arm"]
    covered = np.zeros(case["COV"].shape[1], bool)
    chosen = []
    for _ in range(min(k, len(case["POS"]))):
        pxy = torch.as_tensor(case["POS"], dtype=torch.float32).clone(); pxy[:, 1] /= 4.0
        with torch.no_grad():
            s = PLANNER(torch.as_tensor(planner_features(case, covered)), case["COV"], pxy,
                        torch.tensor([half_x, lift/2.0], dtype=torch.float32))
        if chosen:
            s = s.masked_fill(torch.tensor(np.isin(np.arange(len(s)), chosen)), -1e9)
        a = int(s.argmax()); chosen.append(a); covered |= case["COV"][a]
    return chosen


def greedy_stations(case, k=K_STATIONS):
    """Coverage search weighted by expected utility. The baseline the planner is compared to."""
    Cm, w = case["COV"], case["util"]
    covered = np.zeros(Cm.shape[1], bool)
    S = []
    for _ in range(k):
        gain = ((Cm & ~covered)*w).sum(axis=1)
        if S:
            gain[S] = -1
        b = int(gain.argmax())
        if gain[b] <= 0:
            break
        S.append(b); covered |= Cm[b]
    return S


def sweep_remaining(case, S):
    """One stop for each fruit the first pass left behind.

    A fruit can sit inside the robot's reach and still be missed: the only positions that see it
    add nothing else, so any rule that stops when the marginal gain reaches zero walks past it.
    These are the most expensive fruit on the tree — a stop apiece, about 35 s against 25 s for
    the rest — and the only part of the shortfall that belongs to the planner rather than to the
    detector or the arm. Whether to run the stage is the grower's call; a farm short of pickers
    is trading cheap machine time for scarce human time.
    """
    Cm = case["COV"]
    covered = Cm[S].any(axis=0) if S else np.zeros(Cm.shape[1], bool)
    reach = Cm.any(axis=0)
    extra = []
    for j in np.where(reach & ~covered)[0]:
        cand = [c for c in np.where(Cm[:, j])[0] if c not in S and c not in extra]
        if not cand:
            continue
        b = int(max(cand, key=lambda c: (Cm[c] & ~covered).sum()))
        extra.append(b); covered |= Cm[b]
    return extra


def stations(case, k=K_STATIONS, chooser="planner", sweep=SWEEP):
    """Stops for one tree, and where the first stage ended."""
    S = decode_stations(case, k) if chooser == "planner" else greedy_stations(case, k)
    n1 = len(S)
    if sweep:
        S = S + sweep_remaining(case, S)
    return S, n1


# --- picks -----------------------------------------------------------------------------------
def move_cost_to_each(POS, COV, pos):
    if pos is None:
        return np.zeros(COV.shape[1])
    per = np.abs(POS[:, 0]-pos[0])/E.TRAVEL + np.abs(POS[:, 1]-pos[1])/E.LIFT_SPEED
    return np.where(COV, per[:, None], np.inf).min(axis=0)


def policy_features(st, POS, COV, pos, cycle=E.PICK_SECONDS):
    """Observations, the outcome model's posterior, and the cost of getting to each fruit.

    `features()` is what fills `st.proba`, so it has to run before the posterior is read.
    Folding the two into one expression leaves the rate column computed from a zero posterior —
    a constant once scaled, and the policy then stops on every tree.
    """
    obs = st.features()
    mv = move_cost_to_each(POS, COV, pos)
    mv = np.where(np.isfinite(mv), mv, 1e3)
    u = st.proba @ E.UTIL_VEC
    return np.hstack([obs, st.proba, np.stack([mv, u/(mv + cycle)], axis=1)])


def _next_by_rule(st, live, threshold):
    u = st.utilities()
    ok = live & (u >= threshold) if threshold > 0 else live
    if not ok.any():
        return None
    return int(np.where(ok, u, -np.inf).argmax())


def _next_by_policy(case, st, S, live, pos):
    half_x, lift = case["arm"]
    ctx = torch.tensor([0.0 if pos is None else pos[0],
                        0.0 if pos is None else pos[1]/4.0,
                        half_x, lift/2.0, 1.0, st.alive.mean()], dtype=torch.float32)
    x = torch.as_tensor((policy_features(st, case["POS"], case["COV"], pos) - FMU)/FSD,
                        dtype=torch.float32)
    with torch.no_grad():
        sc = POLICY.scores(x, live.copy(), ctx)
    a = int(sc.argmax())
    return None if a == case["geom"].n else a


def plan_tree(case, k=K_STATIONS, threshold=THRESHOLD, chooser="planner", sweep=SWEEP,
              picker="rule"):
    """The plan for one tree: where to stop, what to take, in what order, and how long.

    `picker="rule"` takes the highest expected utility available, which is the shipping
    configuration. `picker="policy"` uses the trained pick policy, which reaches a different
    operating point — about two thirds of the fruit in half the time, with a twentieth of the
    knocked neighbours. Which one a grower wants depends on whether the block has to be cleared
    or the hour has to be filled; the report gives both.
    """
    geom, Pp, Cp = case["geom"], case["POS"], case["COV"]
    S_all, n1 = stations(case, k, chooser, sweep)
    S = sorted(S_all, key=lambda r: Pp[r][0])
    cov = Cp[S].any(axis=0)

    st = E.TreeState(geom)
    rows, pos, secs, travel = [], None, 0.0, 0.0
    while True:
        live = cov & st.alive
        if not live.any():
            break
        j = (_next_by_policy(case, st, S, live, pos) if picker == "policy"
             else _next_by_rule(st, live, threshold))
        if j is None:
            break
        r = min((rr for rr in S if Cp[rr, j]),
                key=lambda rr: E.move_seconds(pos, tuple(Pp[rr])))
        p = tuple(Pp[r])
        mv = E.move_seconds(pos, p)
        travel += mv; secs += mv + E.PICK_SECONDS; pos = p
        pr = st.probabilities(j)
        rows.append(dict(step=len(rows)+1, stop=S.index(r)+1,
                         x=round(p[0], 3), mast=round(p[1], 3),
                         fruit=int(geom.ids[j]), height=round(float(geom.P[j, 1]), 3),
                         p_success=round(float(pr[E.CLASSES.index("SUCCESS")]), 4),
                         exp_utility=round(float(pr @ E.UTIL_VEC), 4),
                         secs=round(secs, 1)))
        st.remove(j)

    plan = pd.DataFrame(rows)
    col = {c: i for i, c in enumerate(E.CLASSES)}
    P = (np.array([st.probabilities(int(np.where(geom.ids == f)[0][0]))
                   for f in plan.fruit]) if len(plan) else np.zeros((0, len(E.CLASSES))))
    # Kept in .attrs, not as an attribute: pandas warns on the latter, and this function is
    # called a hundred and twenty times in 06 alone.
    plan.attrs["summary"] = dict(
        tree=case["tid"], stops=len(S), stage1_stops=n1, sweep_stops=len(S_all) - n1,
        on_tree=int(geom.n), robot_reach=case["robot_reach"], in_plan=int(cov.sum()),
        attempts=len(plan), seconds=round(secs, 1), travel=round(travel, 1),
        left_reachable=int((st.alive & Cp.any(axis=0)).sum()),
        exp_success=round(float(plan.p_success.sum()), 3) if len(plan) else 0.0,
        exp_utility=round(float(plan.exp_utility.sum()), 3) if len(plan) else 0.0,
        chooser=chooser, picker=picker, k=k, threshold=threshold, sweep=sweep)
    return plan


def plan_summary(tids, **kw):
    """One row per tree. What every downstream notebook aggregates."""
    return pd.DataFrame([plan_tree(tree_case(t), **kw).attrs["summary"] for t in tids])

# --- carrying a plan out ----------------------------------------------------------------------
def realise_tree(case, seed=0, k=K_STATIONS, threshold=THRESHOLD, chooser="planner",
                 sweep=SWEEP, picker="rule"):
    """Plan the tree and carry it out, sampling outcomes instead of taking expectations.

    `plan_tree` reports what the outcome model expects; this reports one draw from it. The
    difference matters in two places. A knocked neighbour is actually removed here, so the rest
    of the tree is planned around a canopy that has changed — the expectation version leaves
    every neighbour standing. And repeating with different seeds gives the spread, which is what
    a paired comparison needs and a single expected value cannot supply.

    Outcomes are drawn from the model, not from the physics. `04_fidelity` re-runs the same
    plans against MuJoCo and reports how far apart the two are.
    """
    rng = np.random.default_rng(seed)
    geom, Pp, Cp = case["geom"], case["POS"], case["COV"]
    S_all, n1 = stations(case, k, chooser, sweep)
    S = sorted(S_all, key=lambda r: Pp[r][0])
    cov = Cp[S].any(axis=0)

    st = E.TreeState(geom)
    rows, pos, secs, travel = [], None, 0.0, 0.0
    while True:
        live = cov & st.alive
        if not live.any():
            break
        j = (_next_by_policy(case, st, S, live, pos) if picker == "policy"
             else _next_by_rule(st, live, threshold))
        if j is None:
            break
        r = min((rr for rr in S if Cp[rr, j]),
                key=lambda rr: E.move_seconds(pos, tuple(Pp[rr])))
        p = tuple(Pp[r])
        mv = E.move_seconds(pos, p)
        travel += mv; secs += mv + E.PICK_SECONDS; pos = p

        pr = st.probabilities(j)
        outcome = E.CLASSES[int(rng.choice(len(E.CLASSES), p=pr))]
        st.remove(j)

        knocked = None
        if outcome == "NEIGHBOR_KNOCKED":
            d = np.where(st.alive, geom.DIST[j], np.inf)
            nb = int(d.argmin())
            if np.isfinite(d[nb]) and rng.random() < E.DETACH_SHARE:
                st.remove(nb); knocked = int(geom.ids[nb])

        rows.append(dict(step=len(rows)+1, stop=S.index(r)+1, fruit=int(geom.ids[j]),
                         p_success=round(float(pr[E.CLASSES.index("SUCCESS")]), 4),
                         exp_utility=round(float(pr @ E.UTIL_VEC), 4),
                         outcome=outcome, knocked=knocked, secs=round(secs, 1)))

    log = pd.DataFrame(rows)
    got = log.outcome.value_counts().to_dict() if len(log) else {}
    log.attrs["summary"] = dict(
        tree=case["tid"], seed=seed, stops=len(S), stage1_stops=n1,
        sweep_stops=len(S_all) - n1, on_tree=int(geom.n),
        robot_reach=case["robot_reach"], in_plan=int(cov.sum()),
        attempts=len(log), seconds=round(secs, 1), travel=round(travel, 1),
        premium=int(got.get("SUCCESS", 0)),
        knocked=int(log.knocked.notna().sum()) if len(log) else 0,
        utility=round(float(sum(E.UTIL_VEC[E.CLASSES.index(o)] for o in log.outcome)), 3)
        if len(log) else 0.0,
        left_reachable=int((st.alive & Cp.any(axis=0)).sum()),
        chooser=chooser, picker=picker, k=k, threshold=threshold, sweep=sweep)
    return log


def realise_summary(tids, seeds=range(30), **kw):
    """One row per tree and seed. What the paired comparisons in 03 aggregate."""
    cases = {t: tree_case(t) for t in tids}
    return pd.DataFrame([realise_tree(cases[t], seed=s, **kw).attrs["summary"]
                         for s in seeds for t in tids])
