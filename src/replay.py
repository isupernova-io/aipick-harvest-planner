"""Render a whole tree, and a plan being carried out on it.

Nothing here is invented. The canopy has no trunk, no scaffold branches and no leaves, because
the model has none either -- what it has is a spur, a five-segment stalk and a fruit, and that
is what gets drawn. The robot is the same: a reach volume the width of the arm and the height
of the mast, its axes, and the stop it is standing at. Every line on screen is a number the
model actually carries.

That restraint is the point. A picture with branches in it would say the branches were
modelled, and a reviewer would be right to ask where their stiffness came from.

    import replay
    R = replay.TreeReplay(E.T, tree=23, arm=(0.200, 0.600))
    R.still("tree.png")                              # the canopy, settled
    R.animate_plan(PLAN, "harvest.mp4", fps=30)      # the plan, carried out

The physics runs once, to let the stalks hang. After that the fruit do not move: picking is
shown by fading a fruit out, and the robot by moving a mocap marker. Rendering a frame is
`mj_forward` and nothing else, so a few hundred frames take seconds rather than an hour.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import mujoco  # noqa: E402

_spec = importlib.util.spec_from_file_location("gen", _HERE/"generate.py")
G = importlib.util.module_from_spec(_spec); G.__name__ = "gen"
_spec.loader.exec_module(G)

# tree frame has y up, the scene has z up; the mapping is its own inverse
R_T2S = np.array([[1.0, 0, 0], [0, 0, 1], [0, 1, 0]])

FRUIT_RGBA = "0.78 0.16 0.14 1"
PICKED_ALPHA = 0.06          # a picked fruit fades rather than vanishing, so the gap is legible
ROBOT_RGBA = "0.18 0.43 0.30 0.16"
AXIS_LEN = 0.18


def units_for_tree(T, tree, visible_only=True, centre=None):
    """Every fruit on one tree, in scene coordinates, ready for the unit builder."""
    df = T[T.tree == tree]
    if visible_only and "vis_any" in df:
        df = df[df.vis_any]
    if not len(df):
        raise ValueError(f"tree {tree} has no fruit")
    P = df[["x", "y", "z"]].to_numpy(float)
    centre = np.asarray(centre if centre is not None else P.mean(0), float)
    return [dict(name=f"F{i}", abs_pt=R_T2S @ (p - centre),
                 sdir=R_T2S @ np.array([r.sdir_x, r.sdir_y, r.sdir_z], float),
                 dim=float(r.dim), apple_id=int(r.fruit_id))
            for i, (p, (_, r)) in enumerate(zip(P, df.iterrows()))], centre


def _robot_xml(half_x, lift, depth=0.55):
    """The arm as the model knows it: a reach volume, its axes, and where it stands.

    A mocap body, so it can be moved between frames without stepping the physics. There is no
    linkage because the model has no linkage -- reach is a box and travel is a speed.
    """
    hx, hy, hz = half_x, lift/2.0, depth/2.0
    axes = "".join(
        f'\n      <geom type="capsule" fromto="0 0 0  {AXIS_LEN*a[0]:.4g} {AXIS_LEN*a[1]:.4g} '
        f'{AXIS_LEN*a[2]:.4g}" size="0.004" rgba="{c}" contype="0" conaffinity="0" mass="0"/>'
        for a, c in (((1, 0, 0), "0.85 0.25 0.20 0.9"),
                     ((0, 1, 0), "0.20 0.55 0.85 0.9"),
                     ((0, 0, 1), "0.25 0.65 0.35 0.9")))
    return f'''
    <body name="robot" mocap="true" pos="0 0 -5">
      <geom name="reach" type="box" size="{hx:.5g} {hz:.5g} {hy:.5g}" rgba="{ROBOT_RGBA}"
            contype="0" conaffinity="0" mass="0"/>
      <geom name="mastpt" type="sphere" size="0.018" rgba="0.10 0.42 0.18 1"
            contype="0" conaffinity="0" mass="0"/>{axes}
    </body>'''


def tree_scene_xml(units, arm=None, ground=True):
    bodies, spurs, excl = [], [], []
    for i, u in enumerate(units):
        b, s = G.unit_xml(u, prefix=f"F{i}_", mu=G.MU_SKIN, label=False)
        bodies.append("    " + b)
        spurs.append("    " + s)
        for k in range(G.N_SEG):
            excl.append(f'    <exclude body1="F{i}_fruit" body2="F{i}_seg{k}"/>')
        excl.append(f'    <exclude body1="world" body2="F{i}_seg{G.N_SEG-1}"/>')

    floor = ('    <geom name="floor" type="plane" size="6 6 0.1" pos="0 0 -2.2" '
             'rgba="0.96 0.96 0.95 1"/>') if ground else ""
    robot = _robot_xml(*arm) if arm else ""
    return f'''<mujoco model="aipick_tree">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{1.0/G.SIM_HZ:.10g}" gravity="0 0 {-G.GRAVITY}" integrator="Euler"/>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1"
             width="256" height="256"/>
  </asset>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight ambient="0.60 0.60 0.60" diffuse="0.50 0.50 0.50" specular="0.10 0.10 0.10"/>
  </visual>
  <default>
    <geom rgba="0.62 0.47 0.26 1"/>
    <joint type="ball" stiffness="{G.STEM_STIFFNESS}" damping="{G.STEM_DAMPING}"/>
  </default>
  <worldbody>
    <light pos="0 -1.5 2.5" dir="0 0.5 -1"/>
{floor}
{chr(10).join(spurs)}
{chr(10).join(bodies)}{robot}
  </worldbody>
  <contact>
{chr(10).join(excl)}
  </contact>
</mujoco>'''


class TreeReplay:
    """A settled canopy that can be rendered many times cheaply."""

    def __init__(self, T, tree, arm=(0.200, 0.600), width=1280, height=800,
                 visible_only=True, settle=None):
        self.tree = tree
        self.arm = arm
        self.units, self.centre = units_for_tree(T, tree, visible_only)
        self.ids = [u["apple_id"] for u in self.units]
        self.index = {a: i for i, a in enumerate(self.ids)}

        self.m = mujoco.MjModel.from_xml_string(tree_scene_xml(self.units, arm))
        self.d = mujoco.MjData(self.m)
        for _ in range(int(settle if settle is not None else G.OBSERVE_SETTLE)):
            mujoco.mj_step(self.m, self.d)
        self.qpos0 = self.d.qpos.copy()

        # which geoms belong to which fruit, so one can be faded without touching the rest
        self.geoms = {}
        for i in range(len(self.units)):
            g = []
            for nm in [f"F{i}_fruit"] + [f"F{i}_seg{k}" for k in range(G.N_SEG)]:
                bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, nm)
                if bid < 0:
                    continue
                g += list(range(self.m.body_geomadr[bid],
                                self.m.body_geomadr[bid] + self.m.body_geomnum[bid]))
            self.geoms[i] = g
        self.rgba0 = self.m.geom_rgba.copy()

        self.renderer = mujoco.Renderer(self.m, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        P = np.array([u["abs_pt"] for u in self.units])
        self.cam.lookat[:] = P.mean(0)
        self.cam.distance = float(np.ptp(P, axis=0).max()*1.9 + 0.9)
        self.cam.azimuth, self.cam.elevation = 90.0, -8.0

    # --- state ------------------------------------------------------------------------------
    def reset(self):
        self.d.qpos[:] = self.qpos0
        self.m.geom_rgba[:] = self.rgba0
        mujoco.mj_forward(self.m, self.d)
        return self

    def set_picked(self, apple_ids):
        """Fade the fruit that have been taken. The stalk stays: the spur is still there."""
        self.m.geom_rgba[:] = self.rgba0
        for a in apple_ids:
            i = self.index.get(int(a))
            if i is None:
                continue
            for g in self.geoms[i]:
                self.m.geom_rgba[g, 3] = PICKED_ALPHA
        return self

    def set_robot(self, x=None, mast=None, depth=0.0):
        """Move the reach marker. Tree-frame x and mast height, as the plan reports them."""
        b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "robot")
        if b < 0:
            return self
        mid = int(self.m.body_mocapid[b])
        if mid < 0:
            return self
        if x is None:
            self.d.mocap_pos[mid] = (0, 0, -5)          # parked out of frame
        else:
            self.d.mocap_pos[mid] = (float(x) - self.centre[0], float(depth),
                                     float(mast) - self.centre[1])
        return self

    def look(self, azimuth=None, elevation=None, distance=None):
        if azimuth is not None:
            self.cam.azimuth = azimuth
        if elevation is not None:
            self.cam.elevation = elevation
        if distance is not None:
            self.cam.distance = distance
        return self

    # --- rendering --------------------------------------------------------------------------
    def frame(self, overlay=None):
        mujoco.mj_forward(self.m, self.d)
        self.renderer.update_scene(self.d, camera=self.cam)
        img = self.renderer.render()
        return _overlay(img, overlay) if overlay else img

    def still(self, path, overlay=None, **look):
        self.look(**look)
        img = self.frame(overlay)
        _write_png(img, path)
        return path

    def animate_plan(self, plan, path, fps=30, hold=12, travel=10, spin=0.0,
                     utility=True, title=None):
        """The plan carried out, one pick at a time.

        `hold` frames on each pick and `travel` frames when the base moves, so the eye can
        follow. `spin` degrees of camera rotation per frame if the canopy should turn.
        """
        frames, taken, util, prem, knock = [], [], 0.0, 0, 0
        self.reset().set_robot(None)
        base = title or f"tree {self.tree}"

        for _ in range(fps//2):                          # a beat on the untouched canopy
            frames.append(self.frame(_hud(base, "before", 0, 0, 0.0, 0.0) if utility else None))

        pos = None
        for r in plan.itertuples():
            here = (float(r.x), float(r.mast))
            if pos != here:
                for k in range(travel):
                    t = (k + 1)/travel
                    x = here[0] if pos is None else pos[0] + (here[0]-pos[0])*t
                    m = here[1] if pos is None else pos[1] + (here[1]-pos[1])*t
                    self.set_robot(x, m)
                    self.cam.azimuth += spin
                    frames.append(self.frame(
                        _hud(base, f"moving to stop {int(r.stop)}", len(taken), knock,
                             util, float(r.secs)) if utility else None))
                pos = here

            taken.append(int(r.fruit))
            prem += 1
            util += float(r.exp_utility)
            self.set_picked(taken)
            for _ in range(hold):
                self.cam.azimuth += spin
                frames.append(self.frame(
                    _hud(base, f"stop {int(r.stop)}  ·  fruit {int(r.fruit)}  "
                               f"P={r.p_success:.2f}", len(taken), knock, util,
                         float(r.secs)) if utility else None))

        for _ in range(fps):                             # hold the finished tree
            self.cam.azimuth += spin
            frames.append(self.frame(
                _hud(base, "plan complete", len(taken), knock, util,
                     float(plan.secs.iloc[-1])) if utility else None))

        return _write_video(frames, path, fps)


# --- overlay ------------------------------------------------------------------------------
def _hud(title, caption, picked, knocked, utility, secs):
    return [f"{title}", caption, "",
            f"picked        {picked:>5}",
            f"knocked       {knocked:>5}",
            f"utility       {utility:>7.1f}",
            f"elapsed       {secs:>6.0f} s"]


def _overlay(img, lines):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return img
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    try:
        big = ImageFont.truetype("arial.ttf", 26)
        mono = ImageFont.truetype("consola.ttf", 20)
    except Exception:
        big = mono = ImageFont.load_default()
    pad, x, y = 14, 22, 20
    w = 330
    h = pad*2 + 34 + 26*(len(lines) - 1)
    dr.rectangle([x-pad, y-pad, x+w, y+h-pad], fill=(255, 255, 255, 205))
    dr.text((x, y), lines[0], fill=(20, 20, 20), font=big)
    yy = y + 36
    for ln in lines[1:]:
        dr.text((x, yy), ln, fill=(60, 60, 60), font=mono)
        yy += 26
    return np.asarray(im)


# --- output -------------------------------------------------------------------------------
def _write_png(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    except ImportError:
        import imageio.v3 as iio
        iio.imwrite(path, img)
    return path


def _write_video(frames, path, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio
        iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264",
                    macro_block_size=8, quality=8)
        return path
    except Exception as ex:
        seq = path.with_suffix("")
        seq.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            _write_png(f, seq/f"{i:05d}.png")
        print(f"  no video writer ({type(ex).__name__}); wrote {len(frames)} frames to {seq}")
        print(f"  ffmpeg -framerate {fps} -i {seq}/%05d.png -pix_fmt yuv420p {path}")
        return seq
