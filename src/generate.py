"""Dataset generator -- MuJoCo apple harvest.

Runs the pick pipeline over the FRESH manifest and writes one row per (pick, aperture).
Sharded so several processes can work at once; each shard owns its own CSV and resumes from
it, so a killed run loses at most one flush.

    python aipick_mj_generate.py --launch --shards 6              spawn 6 workers
    python aipick_mj_generate.py --shard 0 --shards 6             one shard in this process
    python aipick_mj_generate.py --launch --shards 6 --picks 200  a short trial first
    python aipick_mj_generate.py --merge                          concatenate the shards

Each worker is a separate OS process, not a thread: MuJoCo's renderer holds a GL context that
does not survive a fork, so workers are spawned with subprocess. Shards split the manifest by
row index, so no two workers ever touch the same apple.
"""
import argparse
import zlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# The renderer needs a GL backend chosen before mujoco is imported.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pandas as pd
import mujoco

# One root, resolved from the environment so a checkout runs unmodified. Everything the
# generator reads and writes hangs off it.
ROOT   = Path(os.environ.get("AIPICK_ROOT", Path(__file__).resolve().parent.parent))
SRC    = ROOT/"src"
CROPS  = ROOT/"apple_crops"          # FRESH detections, not redistributed
OUT    = ROOT/"data"

sys.path.insert(0, str(SRC))
import physics as A

MAN = pd.read_csv(CROPS/"manifest.csv")


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


# ---- Carried over unchanged from the original generator -----------------------------------
LATENTS = dict(ripeness=("normal", 0.85, 0.10, 0.55, 1.00),
               stem_stiffness=("lognormal", 1.00, 0.25, 0.50, 2.00),
               stem_pull_force=("lognormal", 1.00, 0.15, 0.60, 1.60),
               spur_break_force=("lognormal", 1.00, 0.15, 0.60, 1.60))
CLUSTER_SIZES, CLUSTER_PROBS = [1, 2, 3], [0.20, 0.50, 0.30]
NEIGHBOUR_MAX_D   = 4.0
APERTURE_RATIO    = [0.85, 1.00, 1.20, 1.45]
APERTURE_MIN_RATIO, APERTURE_MAX_RATIO, APERTURE_QUANT = 0.85, 1.60, 0.002
MOTION_BEND, MOTION_TWIST = 35.0, 25.0
SIZE_EST_ABS      = 0.003
MIN_GRASP_FINGERS = 3
# When the scene is observed and approached: AFTER it has fully settled.
#
# The earlier value of 60 steps was chosen to reproduce PyBullet's mean lean of 21 degrees. That
# 21 was measured right after relax_stalks re-imposed the design pose -- it is not an equilibrium.
# Once PyBullet stopped being the reference there was no reason to match it.
#
# Worse, 60 steps lands in the middle of the pendulum swing. Approaching along an axis measured
# there means the fruit has already dropped by the time the gripper arrives, and the fingers miss.
# That was most of the GRASP_FAILED rate.
#
# What survives at equilibrium is decided by the survey: within a frame, centre distance over the
# radius sum has median 2.28 and the surface gap has median 87 mm, so MOST FRUIT NEVER TOUCH.
# Only 6.9% touch; 12.2% come within a tenth of a diameter.
#   fruit that touch nothing   gravity pulls it vertical -> lean 0, cup it from straight below
#   fruit that touch           the neighbour props it up -> it keeps its lean, cup it along
#                              the stalk (the calyx end points where the spur does)
# The stage-8 pair staged 20 mm apart sat at 32.5 degrees in stable equilibrium: the second case.
OBSERVE_SETTLE    = 2400

# Leaves exist to bring the synthetic cloud's composition onto the measured one. The count was
# fixed in aipick_mj_leaves: at six, the gated fruit share is 0.422 +/- 0.038 against 0.4119.
# Visual only (contype=0 mass=0): a real leaf does not stop a gripper, it is brushed aside.
N_LEAVES   = 6
LEAF_LEN   = (0.030, 0.045)      # semi-major axis (m); real apple leaves run 60-90 mm
LEAF_WID   = (0.015, 0.025)
LEAF_THICK = 0.0004
LEAF_RGBA  = "0.22 0.42 0.16 1"
CLOUD_GATE = 2.5                 # in fruit radii
CAM_W = CAM_H = 1300
CAM_FOV = 12.9
CAM_NEAR, CAM_FAR = 0.05, 6.0
CAM_DIST = 2.4118                # median survey distance from pc_stats
NEIGHBOUR_KNOCK_MM  = 3.0            # a neighbour pushed further than this counts as knocked
REACH_MIN_CLEAR     = 0.50           # corridor clearance as a fraction of the aperture
REACH_MIN_POINTS    = 500            # blocked at this many points inside -- a count, not a distance
OBS_FRUIT_MARGIN    = 1.15           # margin when removing the fruit's own points
SEAT_BLOCK_RATIO    = 1.35           # seating above this means it hung on the tips
CAGE_K, FRUIT_K     = 0.640, 0.500

UTILITY = {"SUCCESS": 1.0, "NO_DETACH": 0.0, "APPROACH_BLOCKED": 0.0, "GRASP_FAILED": -0.1,
           "STEM_PULL": -0.3, "STALK_SNAP": -0.3, "SPUR_BREAK": -0.3, "NEIGHBOR_KNOCKED": -0.5}

SEED = 20260810

R_EQ, FINGER_R = 0.028750, 0.0035


_FROZEN = dict(AZ_F_MAX=A.AZ_F_MAX, AZ_BEND_MAX=A.AZ_BEND_MAX, AZ_TWIST_MAX=A.AZ_TWIST_MAX,
               STEM_PULL_FORCE=A.STEM_PULL_FORCE, SPUR_BREAK_FORCE=A.SPUR_BREAK_FORCE)
STEM_STIFFNESS_0, AZ_TWIST_STIFFNESS_0 = 0.030, 0.021      # the values the scene was built with

def _draw(rng, spec):
    kind, mean, sd, lo, hi = spec
    v = rng.normal(mean, sd) if kind == "normal" else float(np.exp(rng.normal(np.log(mean), sd)))
    return float(np.clip(v, lo, hi))

def draw_latents(rng):
    """Conditional draw: stem < AZ < spur. About a tenth of independent draws describe a fruit
    whose stalk would tear the branch off first, which is not an apple."""
    for _ in range(50):
        rip = _draw(rng, LATENTS["ripeness"])
        f_az  = 30.0/rip
        pull  = _FROZEN["STEM_PULL_FORCE"] *_draw(rng, LATENTS["stem_pull_force"])
        spur  = _FROZEN["SPUR_BREAK_FORCE"]*_draw(rng, LATENTS["spur_break_force"])
        if pull < f_az < spur:
            stiff = _draw(rng, LATENTS["stem_stiffness"])
            return dict(lat_ripeness=round(rip, 4), lat_az_f_max=round(f_az, 3),
                        lat_stem_stiffness=round(STEM_STIFFNESS_0*stiff, 5),
                        lat_stem_pull_force=round(pull, 3),
                        lat_spur_break_force=round(spur, 3), _stiff_mult=stiff)
    raise RuntimeError("no valid latent draw in 50 tries")

def apply_latents(model, lat):
    """Verdict constants are module globals; stalk stiffness lives in model.jnt_stiffness."""
    rip = lat["lat_ripeness"]
    A.AZ_F_MAX         = lat["lat_az_f_max"]
    A.AZ_BEND_MAX      = np.deg2rad(45)/rip
    A.AZ_TWIST_MAX     = np.deg2rad(60)/rip
    A.STEM_PULL_FORCE  = lat["lat_stem_pull_force"]
    A.SPUR_BREAK_FORCE = lat["lat_spur_break_force"]
    k = lat["_stiff_mult"]
    for name, base in (("j_in1", STEM_STIFFNESS_0), ("j_in2", STEM_STIFFNESS_0),
                       ("j_in3", STEM_STIFFNESS_0), ("j_in4", STEM_STIFFNESS_0),
                       ("j_az_t", AZ_TWIST_STIFFNESS_0)):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j >= 0: model.jnt_stiffness[j] = base*k
    # Abscission bending (j_az_x/y) is designed with zero stiffness -- leave it alone

def restore_latents():
    for k, v in _FROZEN.items(): setattr(A, k, v)


def euler_dir(rx, ry, rz):
    """The +X column of the rotation matrix. The +Z column gives the camera ray instead -- a
    trap the original source comments warn about."""
    cz, sz, cy, sy = np.cos(rz), np.sin(rz), np.cos(ry), np.sin(ry)
    v = np.array([cz*cy, sz*cy, -sy])
    return v/np.linalg.norm(v)

def cam_to_world(v):
    v = np.asarray(v, float)
    return np.array([v[0], -v[2], v[1]])

def geom_of(row):
    sd = cam_to_world(euler_dir(float(row["rot_x"]), float(row["rot_y"]), float(row["rot_z"])))
    return dict(id=str(row["apple_id"]),
                loc=np.array([row["loc_x"], row["loc_y"], row["loc_z"]], float),
                dim=float(row["dim"]), sdir=sd/np.linalg.norm(sd))

def neighbours_of(row, k, man=MAN):
    if k <= 0: return []
    same = man[(man.frame == row["frame"]) & (man.apple_id != row["apple_id"])]
    if not len(same): return []
    loc = np.array([row["loc_x"], row["loc_y"], row["loc_z"]], float)
    d = np.linalg.norm(same[["loc_x", "loc_y", "loc_z"]].to_numpy(float) - loc, axis=1)/float(row["dim"])
    return [same.iloc[int(i)] for i in np.argsort(d)[:k] if d[i] <= NEIGHBOUR_MAX_D]

def _raw_units(row, k):
    """Assembled with the abscission point at the origin; cluster_units shifts it."""
    g0 = geom_of(row)
    units = [dict(name="A", abs_pt=np.zeros(3), sdir=g0["sdir"], dim=g0["dim"], apple_id=g0["id"])]
    for i, nrow in enumerate(neighbours_of(row, k)):
        g = geom_of(nrow)
        units.append(dict(name=f"N{i}", abs_pt=cam_to_world(g["loc"] - g0["loc"]),
                          sdir=g["sdir"], dim=g["dim"], apple_id=g["id"]))
    return units

def cluster_units(row, k):
    """Shift the whole cluster so the target fruit settles at the origin.

    The approach rises along world z at x=y=0. With the abscission point at the origin, a tilted
    stalk pushes the fruit sideways and the gripper goes to empty space: 28.4 mm of lateral
    offset against a 7.6 mm basket radius. Only the nearest finger touches, which reads as
    GRASP_FAILED, and it was half the failures on picks with no neighbour at all.

    The offset is measured, not derived: the scene is built once, settled and read. A closed
    form would not match, because settling is what decides where the fruit ends up. This calls
    scene_for, which Python resolves at call time, so only the call order matters.
    """
    us = _raw_units(row, k)
    dim = float(row["dim"])
    m = mujoco.MjModel.from_xml_string(scene_for(us, aperture_from(1.0, dim), dim_true=dim))
    d = mujoco.MjData(m)
    settle_scene(m, d)          # holding the palm -- see settle_scene
    off = np.array(d.body("fruit").xpos, float)
    for u in us:
        u["abs_pt"] = np.asarray(u["abs_pt"], float) - off
    return us


# scene_for is not defined yet here, so check with the pre-shift layout (_raw_units)
def aperture_from(ratio, dim_est):
    a = round(ratio*dim_est/APERTURE_QUANT)*APERTURE_QUANT
    return float(np.clip(a, APERTURE_MIN_RATIO*dim_est, APERTURE_MAX_RATIO*dim_est))

def gripper_dims(aperture):
    r = aperture/2
    mount_r = 0.38*r
    return dict(MOUNT_R=mount_r, LOWER_LEN=1.40*r, UPPER_LEN=0.96*r,
                BASE_R=mount_r + FINGER_R + 0.003, PALM_H=1.47*r)

def run_pick(row, rng, build_scene, observe, harvest_once):
    """One pick, one row per aperture.

    build_scene(units, gripper)  builds the scene XML
    observe(model, data, units)  one observation vector
    harvest_once(model, data, units, aperture)  approach, grasp, harvest -> dict(code, ...)

    They are arguments because physics.py already owns the physics calls; this function is
    responsible only for ordering and recording -- the same boundary the original drew.
    """
    lat = draw_latents(rng)
    k   = int(rng.choice(CLUSTER_SIZES, p=CLUSTER_PROBS)) - 1
    units = _raw_units(row, k)

    dim_true = float(row["dim"])
    dim_est  = float(max(dim_true + rng.normal(0, SIZE_EST_ABS), 0.020))   # one look only

    # The observation does not depend on aperture: it is the scene at pick time, made once
    m0 = mujoco.MjModel.from_xml_string(build_scene(units, gripper_dims(aperture_from(1.0, dim_est))))
    d0 = mujoco.MjData(m0)
    apply_latents(m0, lat)
    obs = observe(m0, d0, units)

    order = list(range(len(APERTURE_RATIO)))
    rng.shuffle(order)                       # the last one is the aperture actually committed
    rows = []
    for t_i, ai in enumerate(order):
        ratio    = APERTURE_RATIO[ai]
        aperture = aperture_from(ratio, dim_est)
        is_commit = (t_i == len(order) - 1)

        m = mujoco.MjModel.from_xml_string(build_scene(units, gripper_dims(aperture)))
        d = mujoco.MjData(m)
        apply_latents(m, lat)
        res = harvest_once(m, d, units, aperture)

        r = dict(apple_id=str(row["apple_id"]), frame=int(row["frame"]),
                 aperture_ratio=ratio, aperture_mm=round(aperture*1000, 2),
                 committed=bool(is_commit), code_raw=res["code"], code=res["code"],
                 n_grasp_fingers=res.get("n_fingers", 0),
                 seat_x_palm=res.get("seat_x_palm", 1.0),
                 secs=res.get("secs", 0.0),
                 diag_dim_true_mm=round(dim_true*1000, 2),
                 diag_size_est_err_mm=round((dim_est-dim_true)*1000, 2),
                 diag_fit_ratio=round(aperture/dim_true, 4))
        r.update({f"obs_{kk}": float(v) for kk, v in obs.items()})
        r["obs_dim_est"] = dim_est
        r.update({kk: v for kk, v in lat.items() if not kk.startswith("_")})
        rows.append(r)

    restore_latents()
    return rows


import numpy as np

# ---- constants: frozen values, and the two that had to be corrected ----
R_EQ, OBLATE_Z = 0.028750, 0.88
R_POL          = R_EQ*OBLATE_Z
DIM_H          = 2*R_EQ
FRUIT_MASS     = 0.0735807337962370
FRUIT_INERTIA  = (2.15835e-5, 2.15835e-5, 2.43276e-5)
CAVITY_R, CAVITY_DEPTH = 0.40*R_EQ, 0.18*DIM_H
PEDICEL_LEN    = CAVITY_DEPTH*(1.0+1.15)
N_SEG          = 5
SEG_LEN        = PEDICEL_LEN/N_SEG
SEG_MASS       = 0.020                      # as run, not the 0.002 the frozen json records
SEG_INERTIA    = (2e-5, 2e-5, 1e-5)
PEDICEL_R_BASE, PEDICEL_R_TOP = 0.0030, 0.0018
SPUR_R, SPUR_LEN = 0.0012, 0.030
STEM_STIFFNESS, STEM_DAMPING = 0.030, 0.0015
AZ_TWIST_STIFFNESS = STEM_STIFFNESS*0.7     # circular section; the AZ bends freely
MOUNT_R, LOWER_LEN, UPPER_LEN, PALM_H = 0.38*R_EQ, 1.40*R_EQ, 0.96*R_EQ, 1.47*R_EQ
FINGER_R, BASE_H, BEND_ANGLE = 0.0035, 0.008, np.deg2rad(40)
BASE_R         = MOUNT_R+FINGER_R+0.003
PALM_MASS, PALM_INERTIA = 0.300, 1e-2       # gripper + wrist drivetrain, see notebook
MU_SKIN        = 0.45
LABEL_R        = max(0.006, 2*R_EQ*0.11)   # brand sticker, as the original sizes it
# Hertz contact, for drawing the patch at life size rather than as a sticker.
E_APPLE, NU_APPLE = 6.0e6, 0.35     # apple flesh, 3-8 MPa; the polymer pad is GPa, so it
N_TYPICAL         = 3.5             # drops out of E*.  measured normal force per finger
MEANSIZE          = 0.025           # model.stat.meansize, what MuJoCo scales markers by
_R_STAR = 1/(1/FINGER_R + 1/R_EQ)
_E_STAR = E_APPLE/(1 - NU_APPLE**2)
HERTZ_A = (3*N_TYPICAL*_R_STAR/(4*_E_STAR))**(1/3)
SIM_HZ, GRAVITY = 240, 9.81

def _q(ax, a):
    ax=np.asarray(ax,float); ax=ax/np.linalg.norm(ax)
    return np.concatenate([[np.cos(a/2)], np.sin(a/2)*ax])
def _f(v): return " ".join(f"{x:.9g}" for x in np.atleast_1d(v))


def leaf_xml(seed, centre, fruit_r, n=None):
    """Scatter leaves inside the gate sphere. Visual only, so the physics is untouched.

    The count was fixed in aipick_mj_leaves: six brings the gated fruit share closest to the
    measured 0.4119. One leaf is 76 x 49 mm, larger than the fruit, so six cover plenty.
    """
    n = N_LEAVES if n is None else n
    if n <= 0: return ""
    rng = np.random.default_rng(seed)
    gate_r = CLOUD_GATE*fruit_r
    out = []
    for i in range(n):
        while True:
            d = gate_r*rng.random()**(1/3)
            if d > fruit_r*1.25: break
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        pos = np.asarray(centre, float) + u*d
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        a = rng.uniform(*LEAF_LEN); b = rng.uniform(*LEAF_WID)
        out.append(f'    <geom name="leaf{i}" type="ellipsoid" pos="{_f(pos)}" quat="{_f(q)}" '
                   f'size="{a:.6g} {b:.6g} {LEAF_THICK}" contype="0" conaffinity="0" '
                   f'group="2" mass="0" rgba="{LEAF_RGBA}"/>')
    return "\n".join(out)


def unit_xml(u, prefix="", mu=MU_SKIN, label=True):
    """One fruit, five stalk segments and a spur, assembled in that fruit's own stem frame.

    Target and neighbours are built by the same code. An earlier revision built only the target
    with the stage-9 builder, which pins the stalk vertical, and the target's lean died: sdir
    pointed 21 degrees off vertical while the settled lean read 0.9. obs_lean_deg, obs_ax_* and
    obs_height all went constant, so three model features disappeared.

    prefix="" leaves the names bare. aipick_mj3.Rig looks up ax0..ax4, f_az, f_stem, q_fruit and
    the bodies fruit, seg4, palm plus the geom apple by name, so the target must stay bare.
    """
    P = prefix
    sd = np.asarray(u["sdir"], float); sd = sd/np.linalg.norm(sd)
    q  = align_z(sd)
    dim = float(u["dim"]); r_eq = dim/2; r_pol = r_eq*OBLATE_Z
    cav = dim*0.18
    ped = cav*(1.0 + 1.15)
    seg_len = ped/N_SEG
    apt = np.asarray(u["abs_pt"], float) + sd*((r_pol - cav) + ped)
    node_r = np.linspace(PEDICEL_R_BASE, PEDICEL_R_TOP, N_SEG+1)
    seg_r  = (node_r[:-1]+node_r[1:])/2
    scale = (dim/DIM_H)
    mass  = FRUIT_MASS*scale**3
    inert = np.array(FRUIT_INERTIA)*scale**5

    mark = ""
    if label and not P:
        for m_i, azm in enumerate((np.deg2rad(60), np.deg2rad(180), np.deg2rad(300))):
            n = np.array([np.cos(azm), np.sin(azm), 0.0])
            ax = np.cross([0, 0, 1.0], n); ang = np.arccos(np.clip(n[2], -1, 1))
            lq = _q(ax, ang) if np.linalg.norm(ax) > 1e-9 else np.array([1., 0, 0, 0])
            base = n*r_eq*1.004
            mark += (f'''
        <geom name="lab{m_i}r" type="cylinder" pos="{_f(base)}" quat="{_f(lq)}"
              size="{LABEL_R*(1.18 if m_i==0 else 1.0):.6g} 0.00018" contype="0" conaffinity="0"
              group="2" mass="0" rgba="0.70 0.16 0.15 1"/>
        <geom name="lab{m_i}" type="cylinder" pos="{_f(base + n*0.0002)}" quat="{_f(lq)}"
              size="{LABEL_R*0.78:.6g} 0.00018" contype="0" conaffinity="0"
              group="2" mass="0" rgba="0.98 0.97 0.93 1"/>''')

    site = f'<site name="s_stem" pos="0 0 {(r_pol-cav):.9g}" size="0.0008" rgba="1 0 0 1"/>' if not P else ""
    inner = f'''<body name="{P}fruit" pos="0 0 {-(r_pol-cav):.9g}">
          {site}
          <inertial pos="0 0 0" mass="{mass:.9g}" diaginertia="{_f(inert)}"/>
          <geom name="{P}apple" type="ellipsoid" size="{_f([r_eq,r_eq,r_pol])}"
                condim="4" priority="1" friction="{mu} 0.005 0"
                rgba="{"0.75 0.18 0.16 1" if not P else "0.70 0.20 0.18 1"}"/>{mark}
        </body>'''
    for k in range(N_SEG):
        if k == N_SEG-1:
            s_az = f'<site name="s_az" pos="0 0 0" size="0.0006" rgba="1 0.4 0 1"/>' if not P else ""
            joints = (f'{s_az}'
                      f'<joint name="{P}az_x" type="hinge" axis="1 0 0" stiffness="0" damping="{STEM_DAMPING}"/>'
                      f'<joint name="{P}az_y" type="hinge" axis="0 1 0" stiffness="0" damping="{STEM_DAMPING}"/>'
                      f'<joint name="{P}az_t" type="hinge" axis="0 0 1" stiffness="{AZ_TWIST_STIFFNESS}" damping="{STEM_DAMPING}"/>')
            pos = f'{_f(apt)}" quat="{_f(q)}'
        else:
            joints = f'<joint name="{P}in{k+1}"/>'
            pos = f'0 0 {-seg_len:.9g}'
        vis = (f'<geom name="{P}v{k}" type="cylinder" fromto="0 0 0  0 0 {-seg_len:.9g}" '
               f'size="{SPUR_R}" contype="0" conaffinity="0" group="2" rgba="0.30 0.22 0.14 1"/>') if not P else ""
        inner = f'''<body name="{P}seg{k}" pos="{pos}">
          {joints}
          <inertial pos="0 0 {-seg_len/2:.9g}" mass="{SEG_MASS}" diaginertia="{_f(SEG_INERTIA)}"/>
          <geom name="{P}g{k}" type="cylinder" fromto="0 0 0  0 0 {-seg_len:.9g}"
                size="{seg_r[k]:.9g}" group="3" rgba="0 0 0 0"/>{vis}
          {inner}
        </body>'''
    mid = apt + sd*(SPUR_LEN/2)
    spur = (f'<geom name="{P}spur" type="cylinder" pos="{_f(mid)}" quat="{_f(q)}" '
            f'size="{SPUR_R} {SPUR_LEN/2}" rgba="0.30 0.22 0.14 1"/>')
    return inner, spur

def gripper_xml(g, finger_kp=0.6):
    fingers = ""
    for j in range(3):
        phi = np.deg2rad(120*j); w = np.array([-np.sin(phi), np.cos(phi), 0.0])
        mount = [g["MOUNT_R"]*np.cos(phi), g["MOUNT_R"]*np.sin(phi), BASE_H/2]
        lo_c = "0.16 0.42 0.85 1" if j == 0 else "1 0.6 0.1 1"
        up_c = "0.24 0.52 0.92 1" if j == 0 else "1 0.66 0.18 1"
        fingers += f'''
      <body name="lo{j}" pos="{_f(mount)}" quat="{_f(_q(w, BEND_ANGLE))}">
        <joint name="fj{j}" type="hinge" axis="{_f(-w)}" range="0 1.2" damping="0.002" armature="1e-5"/>
        <geom name="glo{j}" type="capsule" fromto="0 0 0  0 0 {g["LOWER_LEN"]:.9g}" size="{FINGER_R}"
              mass="0.004" rgba="{lo_c}"/>
        <body name="up{j}" pos="0 0 {g["LOWER_LEN"]:.9g}" quat="{_f(_q(w, -BEND_ANGLE))}">
          <geom name="gup{j}" type="capsule" fromto="0 0 0  0 0 {g["UPPER_LEN"]:.9g}" size="{FINGER_R}"
                mass="0.003" rgba="{up_c}"/>
        </body>
      </body>'''
    return fingers

FINGER_CLOSE_0 = 0.32

def _touch_angle(aperture, dim_true):
    g = gripper_dims(aperture)
    reach = (dim_true/2 - g["MOUNT_R"])/max(g["UPPER_LEN"], 1e-9)
    return float(np.arccos(np.clip(reach, -1.0, 1.0)))

# 여유가 음수(−0.549 rad)로 나온다. _touch_angle 이 실제 접촉각의 대리값이지 정확한 값이
# 아니라는 뜻이다 — 손끝이 표면에 닿는 각은 손가락이 두 마디로 꺾여 있어 닫힌 형태가 없다.
# 대리값이라도 쓰는 이유는 둘이다: ①개구에 대해 단조 증가한다 ②기준 크기에서 원래 0.32 와
# 정확히 같다. 상수를 새로 발명하지 않고 기존 값에 기하를 얹은 것이다.

FINGER_MARGIN = FINGER_CLOSE_0 - _touch_angle(2*R_EQ, 2*R_EQ)   # reference: aperture = fruit diameter


def finger_close_for(aperture, dim_true):
    """이 개구에서 손끝이 과일에 닿는 폐합각."""
    return float(np.clip(_touch_angle(aperture, dim_true) + FINGER_MARGIN, 0.10, 1.10))


def settle_scene(model, data, steps=None):
    """팜을 제자리에 붙잡은 채 씬을 정착시킨다.

    ★ 팜은 freejoint 자유 바디다. 서보를 만들기 전에 그냥 mj_step 을 돌리면 구동이 0 이라
    **팜이 자유낙하한다** — 2,400스텝이면 10초, 씬 밖으로 떨어진다. 그 뒤 서보가 생겨도
    되돌리지 못하거나 큰 속도로 도착하고, 손가락은 사과 근처에 있지도 않다.
    지금까지 GRASP_FAILED 로 세던 것의 대부분이 사과를 못 잡은 게 아니라 **팜이 제자리에
    없었던 것**이었다. 착좌 배수가 1.0 에서 1.77 까지 벌어졌던 것이 그 흔적이다.
    """
    n = OBSERVE_SETTLE if steps is None else steps
    mujoco.mj_forward(model, data)
    sv = A.WristServo(model, data, torque_arm=1.0)
    home = np.array(data.xpos[model.body("palm").id], float)
    q0 = np.array([1.0, 0, 0, 0])
    for _ in range(n):
        sv.drive(home, q0, A.APPROACH_FORCE)
        mujoco.mj_step(model, data)
    return sv


def scene_for(units, aperture, mu=MU_SKIN, finger_kp=0.6, palm_z=-0.20,
              dim_true=None, leaf_seed=None):
    """Target, neighbours and a basket sized to the aperture, all from the same builder."""
    g = gripper_dims(aperture)
    A.PALM_H = g["PALM_H"]                       # module global; the approach height comes from it
    if dim_true is not None:                     # the closing angle follows the aperture too
        A.FINGER_CLOSE = finger_close_for(aperture, dim_true)

    bodies, spurs, excl, sens = [], [], [], []
    for i, u in enumerate(units):
        P = "" if i == 0 else f"N{i-1}_"
        b, s = unit_xml(u, prefix=P, mu=mu)
        bodies.append("    "+b); spurs.append("    "+s)
        for k in range(N_SEG):
            excl.append(f'    <exclude body1="{P}fruit" body2="{P}seg{k}"/>')
        excl.append(f'    <exclude body1="world" body2="{P}seg{N_SEG-1}"/>')
        if P: excl.append(f'    <exclude body1="palm" body2="{P}fruit"/>')
    excl += [f'    <exclude body1="palm" body2="{p}{j}"/>' for p in ("lo", "up") for j in range(3)]

    sens.append('    <force name="f_stem" site="s_stem"/>')
    sens.append('    <force name="f_az"   site="s_az"/>')
    sens += [f'    <framezaxis name="ax{k}" objtype="body" objname="seg{k}"/>' for k in range(N_SEG)]
    sens.append('    <framequat name="q_fruit" objtype="body" objname="fruit"/>')
    sens += [f'    <framepos name="N{i}_pos" objtype="body" objname="N{i}_fruit"/>'
             for i in range(len(units)-1)]

    # Leaves go around the target only: the observation is target-centred and the count was
    # calibrated that way. The seed is fixed per apple so all four apertures see the same leaves,
    leaves = leaf_xml(leaf_seed, np.zeros(3), float(units[0]["dim"])/2) if leaf_seed is not None else ""

    act = "\n".join(f'    <position name="a_fj{j}" joint="fj{j}" kp="{finger_kp}"'
                    f' ctrlrange="0 1.2" forcerange="-1.5 1.5"/>' for j in range(3))
    return f'''<mujoco model="aipick_gen">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{1.0/SIM_HZ:.10g}" gravity="0 0 {-GRAVITY}" integrator="Euler"/>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1"
             width="256" height="256"/>
  </asset>
  <visual>
    <global offwidth="{CAM_W}" offheight="{CAM_H}"/>
    <headlight ambient="0.62 0.62 0.62" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>
  </visual>
  <default>
    <geom rgba="0.62 0.47 0.26 1"/>
    <joint type="ball" stiffness="{STEM_STIFFNESS}" damping="{STEM_DAMPING}"/>
  </default>
  <worldbody>
    <light pos="0 0 0.3" dir="0 0 -1"/>
{chr(10).join(spurs)}
{chr(10).join(bodies)}
{leaves}
    <body name="palm" pos="0 0 {palm_z}">
      <freejoint name="wrist"/>
      <inertial pos="0 0 {BASE_H/2}" mass="{PALM_MASS}"
                diaginertia="{PALM_INERTIA} {PALM_INERTIA} {PALM_INERTIA}"/>
      <geom name="gbase" type="cylinder" fromto="0 0 0  0 0 {BASE_H}" size="{g["BASE_R"]:.9g}"
            mass="0" rgba="0.10 0.42 0.18 1"/>{gripper_xml(g, finger_kp)}
    </body>
  </worldbody>
  <contact>
{chr(10).join(excl)}
  </contact>
  <equality>
    <weld name="grasp" body1="palm" body2="fruit" active="false" solref="0.004 1"/>
  </equality>
  <actuator>
{act}
  </actuator>
  <sensor>
{chr(10).join(sens)}
  </sensor>
</mujoco>
'''

def align_z(v):
    v = np.asarray(v, float); v = v/np.linalg.norm(v)
    ax = np.cross([0, 0, 1.0], v); s = np.linalg.norm(ax)
    if s < 1e-9: return np.array([1., 0, 0, 0]) if v[2] > 0 else np.array([0., 1, 0, 0])
    return _q(ax/s, np.arccos(np.clip(v[2], -1, 1)))

# which keeps the counterfactual comparison clean.


def _camera_frame(renderer):
    """The real camera. MjvScene.camera is a stereo pair, so using [0] alone is 34 mm off."""
    sc = renderer.scene
    eye = (np.array(sc.camera[0].pos, float) + np.array(sc.camera[1].pos, float))/2
    fwd = (np.array(sc.camera[0].forward, float) + np.array(sc.camera[1].forward, float))/2
    up  = (np.array(sc.camera[0].up, float) + np.array(sc.camera[1].up, float))/2
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    return eye, fwd, right, np.cross(right, fwd)

def cloud_stats(model, data, renderer, target, fruit_r):
    """Fruit share of the gated points, and the fruit point count.

    Both are things the robot actually knows. A fruit hidden behind leaves gives a poorer size
    estimate and a worse approach axis; a model blind to that plans optimistically.
    """
    gate_r = CLOUD_GATE*fruit_r
    cam = mujoco.MjvCamera()
    cam.lookat[:] = target; cam.distance = CAM_DIST
    cam.azimuth = 90.0; cam.elevation = 0.0
    model.vis.global_.fovy = CAM_FOV
    model.vis.map.znear = CAM_NEAR/model.stat.extent
    model.vis.map.zfar  = CAM_FAR/model.stat.extent

    renderer.enable_depth_rendering()
    renderer.update_scene(data, cam); depth = np.array(renderer.render())
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, cam); seg = np.array(renderer.render())
    renderer.disable_segmentation_rendering()

    eye, fwd, right, up = _camera_frame(renderer)
    h, w = depth.shape
    f = (h/2)/np.tan(np.deg2rad(CAM_FOV)/2)
    yy, xx = np.mgrid[0:h, 0:w]
    dirs = (fwd[None, None, :] + ((xx-(w-1)/2)/f)[..., None]*right[None, None, :]
                               + (-(yy-(h-1)/2)/f)[..., None]*up[None, None, :])
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    pts = eye[None, None, :] + dirs*(depth/np.clip(dirs @ fwd, 1e-6, None))[..., None]

    gid = seg[:, :, 0]
    keep = (gid >= 0) & (depth < CAM_FAR*0.999)
    P, G = pts[keep], gid[keep]
    inside = np.linalg.norm(P - np.asarray(target, float), axis=1) <= gate_r
    P, G = P[inside], G[inside]
    if not len(P): return 0.0, 0
    on = (G == model.geom("apple").id)
    return float(on.mean()), int(on.sum())


def observe(model, data, units, settle_steps=OBSERVE_SETTLE, renderer=None):
    settle_scene(model, data, settle_steps)

    def S(n):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, n)
        a = model.sensor_adr[i]; return data.sensordata[a:a+model.sensor_dim[i]].copy()

    pos = np.array(data.body("fruit").xpos, float)
    top = np.array(data.body("seg4").xpos, float)
    ax  = top - pos; ax /= max(np.linalg.norm(ax), 1e-9)
    lean = float(np.degrees(np.arccos(np.clip(ax[2], -1, 1))))
    sweep = np.array([1.0, 0.0, 0.0])          # scene-fixed axis; same role as fruit_sweep_dir
    sweep = sweep - np.dot(sweep, ax)*ax
    sweep /= max(np.linalg.norm(sweep), 1e-9)

    dim = float(units[0]["dim"])
    others = [np.array(data.body(f"N{i}_fruit").xpos, float) for i in range(len(units)-1)]
    if others:
        rel = [o - pos for o in others]
        dist = [float(np.linalg.norm(r)) for r in rel]
        k = int(np.argmin(dist))
        nu = rel[k]/max(dist[k], 1e-9)
        nd = dist[k]/dim
        bcos, bsin = float(np.dot(nu, sweep)), float(np.dot(nu, np.cross(ax, sweep)))

        # Where the nearest neighbour sits relative to the approach corridor.
        #
        # The corridor test asks whether a neighbour lies below the fruit, inside a narrow
        # cylinder along the approach axis. Bearing alone cannot answer that: it is measured in
        # the plane normal to the axis, so a neighbour 80 mm to the side and one 80 mm directly
        # below produce identical values -- and only the second blocks the approach.
        #
        # That gap cost the outcome model most of its recall on APPROACH_BLOCKED: 19.6% on the
        # six-class argmax, and average precision of 0.32 against 0.94 when the corridor test's
        # own clearance was handed to it. These two numbers are the observable form of that
        # clearance -- the robot detects its neighbours and knows the axis it intends to
        # approach along, so both are measurable before the attempt rather than during it.
        #
        #   nb_along    signed position along the axis, in diameters.
        #               NEGATIVE means below the fruit, which is where the corridor runs
        #   nb_radial   perpendicular distance from the axis, in diameters.
        #               small means directly in the path
        along  = float(np.dot(rel[k], ax))/dim
        radial = float(np.linalg.norm(rel[k] - np.dot(rel[k], ax)*ax))/dim
    else:
        nd, bcos, bsin = 99.0, 0.0, 0.0
        along, radial = 99.0, 99.0        # same sentinel convention as nearest_diam

    # height was dropped. Once the target is shifted to the origin it is constant, and it was
    # only ever a proxy for lean anyway (r = -0.951 against approach_y in PyBullet). lean_deg
    # and ax_* carry it directly. A constant column is ignored by the model but still has to be
    # explained if it stays.
    vis, npts = 0.0, 0
    if renderer is not None:
        vis, npts = cloud_stats(model, data, renderer, pos, dim/2)

    return dict(ax_x=float(ax[0]), ax_y=float(ax[1]), ax_z=float(ax[2]),
                lean_deg=lean, nearest_diam=nd, bearing_cos=bcos, bearing_sin=bsin,
                nb_along=along, nb_radial=radial,
                n_neighbours=len(others), dim=dim,
                visible_frac=vis, n_points=float(npts))


class ContactWatch:
    """Counts contacts on neighbour bodies; the job of watch_reset/_watching_sim_step."""

    def __init__(self, model, watch_bodies, own_bodies):
        self.m = model
        self.watch = {int(model.body(b).id) for b in watch_bodies}
        self.own   = {int(model.body(b).id) for b in own_bodies}
        self.fruit = 0
        self.gripper = 0

    def _body_of(self, geom_id):
        return int(np.atleast_1d(self.m.geom_bodyid[geom_id])[0])

    def step(self, data):
        for i in range(data.ncon):
            c = data.contact[i]
            a, b = self._body_of(c.geom1), self._body_of(c.geom2)
            if   a in self.watch: other = b
            elif b in self.watch: other = a
            else: continue
            if other in self.own: self.fruit += 1
            else:                 self.gripper += 1

def neighbour_moved(model, data, start_pos):
    """How far the neighbour fruit moved since the baseline, in metres. Worst case."""
    if not start_pos: return 0.0
    return max(float(np.linalg.norm(np.array(data.body(b).xpos) - p0))
               for b, p0 in start_pos.items())


# The measured-cloud corridor test was dropped. It swung wildly for the same apple at the same
# aperture (7 blocking points against 1820), which does not fit a deterministic scene. Leaf
# occlusion enters separately as a statistic, so the roles overlapped as well.
#
# Instead the test uses scene bodies only: is a neighbour fruit or its stalk where the gripper
# has to come up? That is also the signal the frame-level planner needs.

# The corridor length is how far the palm actually travels. PALM_H alone is too short: the
# approach starts well below that and strikes neighbours along the way. With a two-neighbour
# test the neighbours sat 72 and 158 mm below the fruit, both outside a 56 mm corridor, and the
# gripper collided with them forty thousand times without the test noticing.
CORRIDOR_RUN    = 0.080      # how far the palm rises (m)
CORRIDOR_MARGIN = 0.010
BLOCK_MIN_BODIES = 1         # one neighbour body in the corridor is already a collision

def approach_blocked_geom(model, data, units, aperture, axis=None):
    """Is a neighbour fruit or stalk inside the approach corridor? Returns (clearance m, count).

    The corridor is a cylinder running from the fruit centre along -axis.
      radius  the swept radius of the basket, so a larger aperture is blocked more often
      length  PALM_H plus the rise, because the gripper starts that far below

    Each body carries its own radius: a centre outside the cylinder can still foul it.
    """
    g = gripper_dims(aperture)
    # The corridor radius is what the gripper sweeps, not the palm disc. The fingers splay by
    # BEND_ANGLE on the way up, so it is far wider. With BASE_R the 0.85 and 1.00 apertures
    # struck neighbours sixty thousand times and still read as clear.
    R_cor = g["MOUNT_R"] + g["LOWER_LEN"]*np.sin(BEND_ANGLE) + FINGER_R
    L_cor = g["PALM_H"] + CORRIDOR_RUN + CORRIDOR_MARGIN

    c = np.array(data.body("fruit").xpos, float)
    if axis is None:
        top = np.array(data.body("seg4").xpos, float)
        axis = (top - c)/max(np.linalg.norm(top - c), 1e-9)
    axis = np.asarray(axis, float); axis = axis/np.linalg.norm(axis)

    worst, n_block = float("inf"), 0
    for i, u in enumerate(units[1:]):
        r_fruit = float(u["dim"])/2
        cand = [(f"N{i}_fruit", r_fruit)]
        cand += [(f"N{i}_seg{k}", PEDICEL_R_BASE) for k in range(N_SEG)]
        for name, r_body in cand:
            try:
                p_b = np.array(data.body(name).xpos, float)
            except KeyError:
                continue
            along = float(np.dot(p_b - c, axis))
            if not (-L_cor <= along <= 0.0):        # outside the approach span
                continue
            radial = float(np.linalg.norm((p_b - c) - along*axis))
            gap = radial - r_body                   # axial distance to the body surface
            worst = min(worst, max(gap, 0.0))
            if gap < R_cor:
                n_block += 1
    return worst, n_block


# ---- Approach and harvest along an arbitrary axis -----------------------------------------
# aipick_mj3's approach_close and act use world z only. Changing the approach direction means
# taking the axis as an argument, so these are axis-aware versions. No new physics: only the
#
# drive targets follow the axis. Note that approach and harvest axes are separate. Going in, the
# gripper tilts to dodge leaves and neighbours; once cupped, bend and twist are given about the
# stalk, because what opens the abscission layer is the angle at the stalk, not where the

def align_q(v):
    v = np.asarray(v, float); v = v/np.linalg.norm(v)
    ax = np.cross([0, 0, 1.0], v); s = np.linalg.norm(ax)
    if s < 1e-9: return np.array([1., 0, 0, 0]) if v[2] > 0 else np.array([0., 1, 0, 0])
    return _q(ax/s, np.arccos(np.clip(v[2], -1, 1)))

def perp_of(v):
    ref = np.array([0, 0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0, 0])
    p = np.cross(v, ref); return p/np.linalg.norm(p)

def approach_close_axis(rig, sv, a_axis, settle=6000, on_step=None):
    """Rise along the approach axis and cup the fruit. Axis-aware approach_close."""
    m, d = rig.m, rig.d
    a = np.asarray(a_axis, float); a = a/np.linalg.norm(a)
    q_ap = align_q(a)
    seat = np.array(d.body("fruit").xpos, float) - a*A.PALM_H
    park = seat - a*0.12

    def hold(p, f, n, phase):
        for _ in range(n):
            sv.drive(p, q_ap, f); mujoco.mj_step(m, d)
            if on_step: on_step(phase)

    d.ctrl[:] = 0.0
    hold(park, A.APPROACH_FORCE, 1200, "park")
    pos = park.copy()
    while np.dot(seat - pos, a) > 0:
        pos = pos + a*0.0002
        hold(pos, A.APPROACH_FORCE, 20, "approach")
        if A.fingers_on_fruit(rig)[1] > A.APPROACH_PUSH_MAX: break
    hold(pos, A.APPROACH_FORCE, 600, "seated")
    A.weld_here(m, d); m.actuator_forcerange[:] = [[-A.FINGER_HOLD, A.FINGER_HOLD]]*3
    for k in range(1, 2401):
        d.ctrl[:] = min(A.FINGER_CLOSE, A.FINGER_CLOSE*k/1200)
        sv.drive(pos, q_ap, A.APPROACH_FORCE); mujoco.mj_step(m, d)
        if on_step: on_step("close")
    hold(pos, A.APPROACH_FORCE, settle, "settle")
    rig.settle(0)
    return pos, q_ap

def act_axis(rig, sv, ref, q_ref, stem, bend=MOTION_BEND, twist=MOTION_TWIST,
             steps=2500, on_step=None):
    """Bend and twist about the stalk. The reference pose is the one at grasp, so a tilted
    approach does not make the wrist lurch."""
    m, d = rig.m, rig.d
    az = d.xpos[m.body("seg4").id].copy()
    st = np.asarray(stem, float); st = st/np.linalg.norm(st)
    pp = perp_of(st)
    B = T = 0.0
    for i in range(steps):
        wb, wt = A.wrist_state(m, d, az, ref, q_ref, st)
        T = min(T + np.deg2rad(twist)/A.SIM_HZ, A.MAX_TWIST, wt + A.LAG_MAX)
        B = min(B + np.deg2rad(bend)/A.SIM_HZ,  A.MAX_BEND,  wb + A.LAG_MAX)
        qb = A.axis_angle_quat(pp, B)
        sv.drive(az + A.qrot(qb, ref - az),
                 A.quat_mul(qb, A.quat_mul(A.axis_angle_quat(st, T), q_ref)),
                 A.WRIST_FORCE_ROLL)
        mujoco.mj_step(m, d)
        if on_step: on_step("act")
        code, where = rig.verdict(rig.measure())
        if code: return code, where, i
    return "NO_DETACH", "none", steps


# ---- Blocked? come back from another direction --------------------------------------------
# A person whose way is blocked changes the angle and tries again. With a single direction a
# blocked pick was simply a failure, and much of the 11% APPROACH_BLOCKED rate was that.
#
# The tilt limit comes from this gripper's geometry, not from the literature: pick an
# unobstructed fruit at a range of angles and see where success stops.

TILT_TRY   = (12.0, 24.0, 36.0)      # tilts to try (degrees)
AZIM_TRY   = 8                       # azimuths swept per tilt
TILT_MAX   = 36.0                    # limit measured in the notebook

def tilted(axis, tilt_deg, azim_rad):
    """Tilt the axis by tilt_deg; azim picks the direction in the plane normal to it."""
    a = np.asarray(axis, float); a = a/np.linalg.norm(a)
    u = perp_of(a); v = np.cross(a, u)
    dirn = np.cos(azim_rad)*u + np.sin(azim_rad)*v
    t = np.deg2rad(tilt_deg)
    return a*np.cos(t) + dirn*np.sin(t)

def find_open_axis(model, data, units, aperture, stem):
    """Find an unblocked approach axis. Returns (axis, tilt degrees, tries)."""
    _, n0 = approach_blocked_geom(model, data, units, aperture, axis=stem)
    if n0 < BLOCK_MIN_BODIES:
        return np.asarray(stem, float), 0.0, 1
    tries = 1
    for tilt in TILT_TRY:
        if tilt > TILT_MAX: break
        for j in range(AZIM_TRY):
            ax = tilted(stem, tilt, 2*np.pi*j/AZIM_TRY)
            _, n = approach_blocked_geom(model, data, units, aperture, axis=ax)
            tries += 1
            if n < BLOCK_MIN_BODIES:
                return ax, tilt, tries
    return np.asarray(stem, float), 0.0, tries      # no angle opens up


def label(code_phys, n_grasp_fingers, unreachable, gripper_hits, fruit_hits,
          seat_x_palm, moved_m):
    """The original generator's label branch, unchanged."""
    code = code_phys
    if n_grasp_fingers < MIN_GRASP_FINGERS and not unreachable:
        code = "GRASP_FAILED"
    elif unreachable or (gripper_hits > 0 and seat_x_palm > SEAT_BLOCK_RATIO):
        code = "APPROACH_BLOCKED"
    elif moved_m*1000 > NEIGHBOUR_KNOCK_MM or fruit_hits > 0:
        if code == "SUCCESS": code = "NEIGHBOR_KNOCKED"
    return code

DEFECT_CODES = ("STEM_PULL", "STALK_SNAP", "SPUR_BREAK")

def to_model_class(code):
    """Into the five classes the model was fitted on. The generator keeps the three damage
    codes apart and they are merged at fitting time."""
    return "DEFECT" if code in DEFECT_CODES else code

# Branch table -- a visual check that the answers match the original


def harvest_once(model, data, units, aperture, row=None,
                 bend=MOTION_BEND, twist=MOTION_TWIST):
    """One approach, grasp and harvest. aipick_mj3 does the physics; this attaches the label.

    `row` is taken for the corridor test. An earlier revision hard-coded unreachable to False,
    so picks whose path was blocked leaked out as GRASP_FAILED -- the gripper struck the
    neighbour sixty thousand times, shoved it 67 mm, and the label still read "failed to grasp".
    Those picks are exactly what APPROACH_BLOCKED exists for.
    """
    t0 = time.time()
    rig = A.Rig.__new__(A.Rig); rig.m, rig.d = model, data
    rig.stem_dir = np.array([0., 0, 1.])
    # A fresh scene is compiled per aperture, so this mjData has not settled. Without settling,
    # the stalk axis below is the design pose and the fruit drops away during the approach.
    # And the palm has to be held while settling or it free-falls (see settle_scene).
    sv = settle_scene(model, data)

    nb = [f"N{i}_fruit" for i in range(len(units)-1)]
    watch = ContactWatch(model, nb, ["fruit"] + [f"seg{k}" for k in range(5)]) if nb else None
    # The baseline is taken after approach and grasp. Measured from pick start it would include
    # the neighbour's own pendulum swing: with no gripper at all it moves 27.3 mm, nine times
    # the 3 mm threshold. Nearly all of a 38% NEIGHBOR_KNOCKED rate was that.
    start = {}

    # Choose the approach axis; if the stalk axis is blocked, tilt and look again
    c_f  = np.array(data.body("fruit").xpos, float)
    top_ = np.array(data.body("seg4").xpos, float)
    stem = (top_ - c_f)/max(np.linalg.norm(top_ - c_f), 1e-9)
    a_axis, tilt_deg, n_tries = find_open_axis(model, data, units, aperture, stem)
    clear_m, nblock = approach_blocked_geom(model, data, units, aperture, axis=a_axis)
    unreachable = nblock >= BLOCK_MIN_BODIES

    def on(_phase):
        if watch: watch.step(data)

    ref, q_ref = approach_close_axis(rig, sv, a_axis, on_step=on)
    start = {b: np.array(data.body(b).xpos, float) for b in nb}   # displacement baseline
    # Contact counts are NOT reset. Displacement mixes in drift, but a contact is unambiguous:
    # striking a neighbour during the approach still counts. The two are timed differently.
    n_fingers, N = A.fingers_on_fruit(rig)
    seat_x_palm = float(np.linalg.norm(ref - np.array(data.body("fruit").xpos, float))/A.PALM_H)

    code, where, steps = act_axis(rig, sv, ref, q_ref, stem, bend=bend, twist=twist, on_step=on)
    moved = neighbour_moved(model, data, start)

    lab = label(code, n_fingers, unreachable=unreachable,
                gripper_hits=watch.gripper if watch else 0,
                fruit_hits=watch.fruit if watch else 0,
                seat_x_palm=seat_x_palm, moved_m=moved)
    return dict(code=lab, code_phys=code, where=where, n_fingers=n_fingers,
                normal_n=N, seat_x_palm=round(seat_x_palm, 3),
                neighbour_moved_mm=round(moved*1000, 2),
                hit_by_fruit=watch.fruit if watch else 0,
                hit_by_gripper=watch.gripper if watch else 0,
                diag_clearance_mm=round(min(clear_m, 9.999)*1000, 2),
                diag_blocking_bodies=int(nblock),
                finger_close=round(float(A.FINGER_CLOSE), 4),
                approach_tilt_deg=round(float(tilt_deg), 1), n_tries=int(n_tries),
                steps=steps, secs=round(time.time()-t0, 2))


def one_pick(row, rng):
    lat = draw_latents(rng)
    k   = int(rng.choice(CLUSTER_SIZES, p=CLUSTER_PROBS)) - 1
    units = cluster_units(row, k)
    dim_true = float(row["dim"])
    dim_est  = float(max(dim_true + rng.normal(0, SIZE_EST_ABS), 0.020))

    leaf_seed = stable_seed(row["apple_id"])   # fixed per apple across apertures
    m0 = mujoco.MjModel.from_xml_string(scene_for(units, aperture_from(1.0, dim_est),
                                                 dim_true=dim_true, leaf_seed=leaf_seed))
    d0 = mujoco.MjData(m0); apply_latents(m0, lat)
    rend = mujoco.Renderer(m0, CAM_H, CAM_W)
    obs = observe(m0, d0, units, renderer=rend)
    del rend

    order = list(range(len(APERTURE_RATIO))); rng.shuffle(order)
    rows = []
    for t_i, ai in enumerate(order):
        ratio = APERTURE_RATIO[ai]; ap = aperture_from(ratio, dim_est)
        m = mujoco.MjModel.from_xml_string(scene_for(units, ap, dim_true=dim_true, leaf_seed=leaf_seed))
        d = mujoco.MjData(m); apply_latents(m, lat)
        res = harvest_once(m, d, units, ap, row=row)
        r = dict(apple_id=str(row["apple_id"]), frame=int(row["frame"]),
                 aperture_ratio=ratio, aperture_mm=round(ap*1000, 2),
                 committed=(t_i == len(order)-1), **res)
        r.update({f"obs_{kk}": float(v) for kk, v in obs.items()})
        r["obs_dim_est"] = dim_est
        r.update({kk: v for kk, v in lat.items() if not kk.startswith("_")})
        rows.append(r)
    restore_latents()
    return pd.DataFrame(rows).sort_values("aperture_ratio")


# ---- Sharded driver -------------------------------------------------------------------------



    """Append rows, keeping the column set fixed.

    An aperture with no neighbour produces one key fewer, so a naive append writes a header
    from the first batch and then a wider row later. pandas then refuses to read the file back.
    """
    global ROW_ORDER
    if not buf:
        return
    df = pd.DataFrame(buf)
    if ROW_ORDER is None:
        ROW_ORDER = list(df.columns)
    df = df.reindex(columns=ROW_ORDER)
    df.to_csv(dest, mode="a", header=not dest.exists(), index=False)


    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT/f"rows_shard{shard:02d}.csv"

    targets = MAN.iloc[shard::n_shards].reset_index(drop=True)
    if picks:
        targets = targets.head(int(np.ceil(picks/n_shards)))

    done = set()
    if dest.exists():
        try:
            prev = pd.read_csv(dest)
            done = set(prev.apple_id.astype(str))
            ROW_ORDER = list(prev.columns)
        except Exception as e:
            print(f"[{shard}] could not read {dest.name}: {e}", flush=True)

    todo = [r for _, r in targets.iterrows() if str(r.apple_id) not in done]
    print(f"[{shard}] {len(todo)} picks to do, {len(done)} already recorded", flush=True)

    rng = np.random.default_rng(seed + shard)
    buf, t0, n = [], time.time(), 0
    for row in todo:
        try:
            buf.extend(one_pick(row, rng).to_dict("records"))
        except Exception as e:
            print(f"[{shard}] {row.apple_id} failed: {type(e).__name__}: {e}", flush=True)
        n += 1
        if len(buf) >= flush_every*len(APERTURE_RATIO):
            _flush(buf, dest); buf = []
            el = time.time() - t0
            print(f"[{shard}] {n}/{len(todo)}  {el/60:.1f} min  "
                  f"eta {el/n*(len(todo)-n)/60:.1f} min", flush=True)
    _flush(buf, dest)
    print(f"[{shard}] done in {(time.time()-t0)/60:.1f} min -> {dest}", flush=True)


ROW_ORDER = None            # column order, fixed on the first flush so appends line up


def _flush(buf, dest):
    """Append rows, keeping the column set fixed.

    An aperture with no neighbour yields one key fewer, so a naive append writes a header from
    the first batch and a wider row later; pandas then refuses to read the file back.
    """
    global ROW_ORDER
    if not buf:
        return
    df = pd.DataFrame(buf)
    if ROW_ORDER is None:
        ROW_ORDER = list(df.columns)
    df = df.reindex(columns=ROW_ORDER)
    df.to_csv(dest, mode="a", header=not dest.exists(), index=False)


def run_shard(shard, n_shards, picks=None, seed=20260810, flush_every=25):
    """Generate this shard's rows, resuming from whatever is already on disk."""
    global ROW_ORDER
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT/f"rows_shard{shard:02d}.csv"

    targets = MAN.iloc[shard::n_shards].reset_index(drop=True)
    if picks:
        targets = targets.head(int(np.ceil(picks/n_shards)))

    done = set()
    if dest.exists():
        try:
            prev = pd.read_csv(dest)
            done = set(prev.apple_id.astype(str))
            ROW_ORDER = list(prev.columns)
        except Exception as e:
            print(f"[{shard}] could not read {dest.name}: {e}", flush=True)

    todo = [r for _, r in targets.iterrows() if str(r.apple_id) not in done]
    print(f"[{shard}] {len(todo)} picks to do, {len(done)} already recorded", flush=True)

    rng = np.random.default_rng(seed + shard)
    buf, t0, n = [], time.time(), 0
    for row in todo:
        try:
            buf.extend(one_pick(row, rng).to_dict("records"))
        except Exception as e:
            print(f"[{shard}] {row.apple_id} failed: {type(e).__name__}: {e}", flush=True)
        n += 1
        if len(buf) >= flush_every*len(APERTURE_RATIO):
            _flush(buf, dest); buf = []
            el = time.time() - t0
            print(f"[{shard}] {n}/{len(todo)}  {el/60:.1f} min  "
                  f"eta {el/n*(len(todo)-n)/60:.1f} min", flush=True)
    _flush(buf, dest)
    print(f"[{shard}] done in {(time.time()-t0)/60:.1f} min -> {dest}", flush=True)


def launch(n_shards, picks=None, seed=20260810):
    """Spawn one process per shard and wait.

    Six is a reasonable default: each worker holds its own MuJoCo model, data and renderer,
    and the renderer is the memory-hungry part at 1300x1300.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    procs = []
    for k in range(n_shards):
        cmd = [sys.executable, os.path.abspath(__file__),
               "--shard", str(k), "--shards", str(n_shards), "--seed", str(seed)]
        if picks:
            cmd += ["--picks", str(picks)]
        log = open(OUT/f"shard{k:02d}.log", "w", encoding="utf-8")
        procs.append((k, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"launched shard {k} (pid {procs[-1][1].pid})", flush=True)

    t0 = time.time()
    fails = []
    for k, p, log in procs:
        rc = p.wait(); log.close()
        if rc != 0:
            fails.append(k)
        print(f"shard {k} exited {rc}", flush=True)
    print(f"\nall shards finished in {(time.time()-t0)/60:.1f} min", flush=True)
    if fails:
        print(f"failed shards: {fails} -- see {OUT}/shardNN.log", flush=True)


def merge():
    """Concatenate the shard CSVs into one dataset."""
    files = sorted(OUT.glob("rows_shard*.csv"))
    if not files:
        print("nothing to merge"); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dest = OUT/"rows_full.csv"
    df.to_csv(dest, index=False)
    print(f"{len(files)} shards -> {len(df)} rows, {df.apple_id.nunique()} apples -> {dest}")
    print("\noutcome classes")
    for k, v in df.code.value_counts().items():
        print(f"  {k:<18} {v:5d}  ({v/len(df)*100:.1f}%)")
    print("\nmodel classes")
    for k, v in df.code.map(to_model_class).value_counts().items():
        print(f"  {k:<18} {v:5d}  ({v/len(df)*100:.1f}%)")
    print("\nSUCCESS by aperture")
    for r, g in df.groupby("aperture_ratio"):
        print(f"  {r:.2f}  {(g.code == 'SUCCESS').mean()*100:.1f}%  (n={len(g)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--launch", action="store_true", help="spawn workers and wait")
    ap.add_argument("--merge", action="store_true", help="concatenate shard CSVs")
    ap.add_argument("--shard", type=int, default=None, help="run this shard in-process")
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--picks", type=int, default=None, help="cap total picks (trial runs)")
    ap.add_argument("--seed", type=int, default=20260810)
    # Positional form, so the PowerShell launcher can call `python aipick_mj_generate.py 0 6`
    ap.add_argument("pos", nargs="*", help="shard n_shards")
    a = ap.parse_args()

    if a.pos and a.shard is None:
        a.shard = int(a.pos[0])
        if len(a.pos) > 1:
            a.shards = int(a.pos[1])

    if a.merge:
        merge()
    elif a.shard is not None:
        run_shard(a.shard, a.shards, picks=a.picks, seed=a.seed)
    elif a.launch:
        launch(a.shards, picks=a.picks, seed=a.seed)
    else:
        ap.print_help()
