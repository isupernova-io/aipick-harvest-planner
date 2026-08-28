"""Rebuild the observation block of an existing dataset.

The physics does not change -- only what is recorded about the scene at pick time. So the picks
are not re-run; each apple's cluster is staged, settled and observed once, and the resulting
obs_* columns replace those in the existing rows.

    python reobserve.py --launch --shards 6
    python reobserve.py --shard 0 --shards 6
    python reobserve.py --merge

Why this exists: APPROACH_BLOCKED depends on whether a neighbour sits inside the approach
corridor, and the observation carried only a bearing measured in the plane normal to the axis --
so a neighbour beside the fruit and one directly below it looked identical. obs_nb_along and
obs_nb_radial encode the missing dimension.

Cost is roughly a fifth of generation: staging and settling only, no approach, grasp or harvest.
"""
import argparse
import zlib, os, subprocess, sys, time
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pandas as pd
import mujoco

AIPICK = Path(os.environ.get("AIPICK_ROOT", r"C:\aipick"))
MJ     = AIPICK/"mujoco"
CROPS  = AIPICK/"apple_crops"
DATA   = MJ/"dataset"
OUT    = DATA/"reobs"

sys.path.insert(0, str(MJ))
import importlib.util
spec = importlib.util.spec_from_file_location("gen", MJ/"aipick_mj_generate.py")
G = importlib.util.module_from_spec(spec); G.__name__ = "gen"
spec.loader.exec_module(G)

def stable_seed(key):
    """A seed that survives process boundaries.

    Python's built-in hash() is salted per process (PYTHONHASHSEED is random by default), so
    hash("000123_45_67") returns a different number in every run. Leaf placement keyed on it is
    reproducible within one generation run and nowhere else -- which surfaced when the
    observations were rebuilt in a separate process and obs_visible_frac came back with the same
    distribution but a correlation of 0.51 against the original values.

    CRC32 is stable across processes, machines and versions. It is not cryptographic, which does
    not matter here: it is being used to spread seeds, not to hide anything.
    """
    return zlib.crc32(str(key).encode()) & 0x7FFFFFFF


MAN = pd.read_csv(CROPS/"manifest.csv")
MAN["apple_id"] = MAN.apple_id.astype(str)


def observe_apple(row, k, rng_seed):
    """Stage this apple's cluster as the original pick saw it, and observe it.

    The cluster size is NOT re-drawn. The generator seeded each pick with SEED plus its index in
    that shard's work list, and the index is not recoverable from the saved rows -- re-drawing
    would stage a different number of neighbours and the new observation would then describe a
    scene the recorded outcome never came from.

    Instead k is taken from the row's own obs_n_neighbours. Neighbour SELECTION is deterministic
    (nearest first, within four diameters), so fixing the count reproduces the original cluster
    exactly. Latents are still drawn, because the scene needs them, but they do not enter the
    observation -- only the pose and the geometry do.
    """
    rng   = np.random.default_rng(rng_seed)
    lat   = G.draw_latents(rng)
    units = G.cluster_units(row, int(k))
    dim   = float(row["dim"])
    leaf  = stable_seed(row["apple_id"])

    m = mujoco.MjModel.from_xml_string(
        G.scene_for(units, G.aperture_from(1.0, dim), dim_true=dim, leaf_seed=leaf))
    d = mujoco.MjData(m)
    G.apply_latents(m, lat)
    rend = mujoco.Renderer(m, G.CAM_H, G.CAM_W)
    obs = G.observe(m, d, units, renderer=rend)
    del rend
    return {f"obs_{k2}": v for k2, v in obs.items()}


def run_shard(shard, n_shards, flush_every=50):
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT/f"obs_shard{shard:02d}.csv"

    # One row per apple: the four aperture rows share an observation, so the first will do.
    rows = pd.read_csv(DATA/"rows_full.csv", usecols=["apple_id", "obs_n_neighbours"])
    rows["apple_id"] = rows.apple_id.astype(str)
    per_apple = rows.drop_duplicates("apple_id").set_index("apple_id").obs_n_neighbours
    ids = pd.Index(per_apple.index)[shard::n_shards]

    done = set()
    if dest.exists():
        try:
            done = set(pd.read_csv(dest).apple_id.astype(str))
        except Exception as e:
            print(f"[{shard}] could not read {dest.name}: {e}", flush=True)
    todo = [i for i in ids if i not in done]
    print(f"[{shard}] {len(todo)} apples to observe, {len(done)} already done", flush=True)

    buf, t0 = [], time.time()
    for n, aid in enumerate(todo, 1):
        sel = MAN[MAN.apple_id == aid]
        if not len(sel):
            continue
        row = sel.iloc[0]
        try:
            rec = observe_apple(row, per_apple[aid], stable_seed(aid))
        except Exception as e:
            print(f"[{shard}] {aid} failed: {type(e).__name__}: {e}", flush=True)
            continue
        rec["apple_id"] = aid
        buf.append(rec)
        if len(buf) >= flush_every:
            pd.DataFrame(buf).to_csv(dest, mode="a", header=not dest.exists(), index=False)
            buf = []
            el = time.time() - t0
            print(f"[{shard}] {n}/{len(todo)}  {el/60:.1f} min  "
                  f"eta {el/n*(len(todo)-n)/60:.1f} min", flush=True)
    if buf:
        pd.DataFrame(buf).to_csv(dest, mode="a", header=not dest.exists(), index=False)
    print(f"[{shard}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def launch(n_shards):
    OUT.mkdir(parents=True, exist_ok=True)
    procs = []
    for k in range(n_shards):
        cmd = [sys.executable, os.path.abspath(__file__), "--shard", str(k),
               "--shards", str(n_shards)]
        log = open(OUT/f"obs_shard{k:02d}.log", "w", encoding="utf-8")
        procs.append((k, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"launched shard {k} (pid {procs[-1][1].pid})", flush=True)
    t0 = time.time()
    for k, p, log in procs:
        rc = p.wait(); log.close()
        print(f"shard {k} exited {rc}", flush=True)
    print(f"\nall shards finished in {(time.time()-t0)/60:.1f} min", flush=True)


def merge():
    """Replace the obs_* block of rows_full.csv with the freshly observed one."""
    files = sorted(OUT.glob("obs_shard*.csv"))
    if not files:
        print("nothing to merge"); return
    O = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    O["apple_id"] = O.apple_id.astype(str)
    O = O.drop_duplicates("apple_id")

    D = pd.read_csv(DATA/"rows_full.csv")
    D["apple_id"] = D.apple_id.astype(str)

    old_obs = [c for c in D.columns if c.startswith("obs_")]
    new_obs = [c for c in O.columns if c.startswith("obs_")]

    # Not every obs_ column comes from observe(). obs_dim_est is drawn once per pick in
    # one_pick() -- it is the robot's noisy size estimate, not a property of the settled scene --
    # so re-observing never produces it. Replacing the whole obs_ block wholesale would delete
    # it. Columns the rebuild does not produce are kept from the original rows.
    keep_old = [c for c in old_obs if c not in new_obs]
    drop_old = [c for c in old_obs if c in new_obs]
    print(f"rebuilt {len(new_obs)} obs columns; keeping {len(keep_old)} the rebuild does not "
          f"produce: {keep_old}")
    print(f"  added: {sorted(set(new_obs) - set(old_obs))}")

    # Sanity check. obs_n_neighbours must be identical -- it was forced. obs_lean_deg should be
    # too, since the staging is deterministic once the cluster is fixed.
    #
    # obs_visible_frac is the exception, and expectedly so. Leaf placement is random by design,
    # and the original dataset seeded it with Python's salted hash() -- so those leaves cannot be
    # reproduced in any later process. The distribution matches (mean 0.358 against 0.357) but
    # the per-apple values do not.
    #
    # This is harmless for the outcome columns: leaves are visual-only geoms (contype=0, mass=0)
    # and the corridor test reads neighbour bodies, so no leaf has ever changed a pick's result.
    # It does mean the occlusion feature and the recorded outcome describe slightly different
    # scenes. With stable_seed in place both files now agree from this run onward.
    chk = D[["apple_id"] + [c for c in old_obs if c in new_obs]].drop_duplicates("apple_id") \
           .merge(O, on="apple_id", suffixes=("_old", "_new"))
    print("\nagreement between old and new observations")
    for c in ("obs_lean_deg", "obs_dim_est", "obs_visible_frac", "obs_n_neighbours"):
        if f"{c}_old" in chk and f"{c}_new" in chk:
            r = np.corrcoef(chk[f"{c}_old"], chk[f"{c}_new"])[0, 1]
            same = (chk[f"{c}_old"].round(4) == chk[f"{c}_new"].round(4)).mean()
            print(f"  {c:<20} r={r:+.4f}  identical {same*100:5.1f}%")

    D = D.drop(columns=drop_old).merge(O, on="apple_id", how="left")
    missing = D[new_obs[0]].isna().sum()
    if missing:
        print(f"\n{missing} rows have no new observation -- dropping them")
        D = D[D[new_obs[0]].notna()]

    dest = DATA/"rows_full_v2.csv"
    D.to_csv(dest, index=False)
    print(f"\n{len(D):,} rows, {D.apple_id.nunique():,} apples -> {dest}")
    print("Written alongside the original; rows_full.csv is untouched.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("pos", nargs="*")
    a = ap.parse_args()
    if a.pos and a.shard is None:
        a.shard = int(a.pos[0])
        if len(a.pos) > 1: a.shards = int(a.pos[1])
    if a.merge: merge()
    elif a.shard is not None: run_shard(a.shard, a.shards)
    elif a.launch: launch(a.shards)
    else: ap.print_help()
