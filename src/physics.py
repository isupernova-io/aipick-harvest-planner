"""Shared measurement and grasp helpers for the MuJoCo port.

Stages 1-3 each defined what they needed inline, which was fine while there were
two copies. This is the third stage that needs the same damage sums and the same
force conventions, and three copies drift: a fix lands in one and not the others.

Everything here was read out of the PyBullet build's definitions and then checked
against poses whose answer is known by hand -- see stage3_damage.ipynb, which is
where the verification lives. Nothing in this file is tuned.
"""

# Bumped whenever this file changes. Notebooks import it by name, so a stale copy
# left behind by a browser download (aipick_mj(1).py) fails silently rather than
# loudly - print this at the top of any notebook and check it against the handoff.
REVISION = "2026-08-08 s9g + gravity feed-forward, approach stops on push"

import mujoco
import numpy as np
from pathlib import Path

# The scene files live beside this module. from_xml_path resolves a bare name against the
# working directory, so calling any of the three loaders below from a notebook one folder up
# used to fail on a file that was sitting right next to the code.
_HERE = Path(__file__).resolve().parent


def _scene(name):
    p = Path(name)
    return str(p if p.is_absolute() else _HERE/p)

AZ_F_MAX=30.0; AZ_BEND_MAX=np.deg2rad(45); AZ_TWIST_MAX=np.deg2rad(60)
STEM_PULL_FORCE=18.0; SPUR_BREAK_FORCE=45.0
N_STEM_SEG=5; INTERNODE_BEND_MAX=np.deg2rad(45)/0.71
BASELINE_MAX_x_MG=3.0; FRUIT_MASS=0.0735807337962370

def az_damage(F,bend,twist):
    return (F/AZ_F_MAX)**2 + (bend/AZ_BEND_MAX)**2 + (twist/AZ_TWIST_MAX)**2
def internode_damage(bend,twist):
    return (bend/INTERNODE_BEND_MAX)**2 + (twist/(N_STEM_SEG-1)/AZ_TWIST_MAX)**2

def quat_mul(a,b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
def quat_conj(q): return np.array([q[0],-q[1],-q[2],-q[3]])

def twist_about_axis(q_rel, axis):
    """swing-twist. MuJoCo quaternion order is (w,x,y,z); PyBullet's was (x,y,z,w)."""
    axis=np.asarray(axis,float); axis=axis/np.linalg.norm(axis)
    q=np.asarray(q_rel,float)
    if q[0] < 0.0: q = -q                  # hemisphere nearest identity
    proj=np.dot(q[1:],axis)*axis
    tw=np.array([q[0],proj[0],proj[1],proj[2]]); n=np.linalg.norm(tw)
    if n<1e-9: return 0.0
    tw/=n
    return 2*np.arctan2(np.dot(tw[1:],axis), tw[0])

class Rig:
    def __init__(self, xml="stalk_scene.xml"):
        self.m=mujoco.MjModel.from_xml_path(_scene(xml)); self.d=mujoco.MjData(self.m)
        self.stem_dir=np.array([0.,0.,1.])
    def S(self,n):
        i=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_SENSOR,n); a=self.m.sensor_adr[i]
        return self.d.sensordata[a:a+self.m.sensor_dim[i]].copy()
    def joint_bends(self):
        ax=[self.S(f"ax{k}") for k in range(5)]          # ax0 = fruit end
        inter=[float(np.arccos(np.clip(np.dot(ax[j],ax[j+1]),-1,1))) for j in range(4)]
        az=float(np.arccos(np.clip(np.dot(ax[4],self.stem_dir),-1,1)))
        return inter, az
    def f_az(self):   return float(np.linalg.norm(self.S("f_az")))
    def f_stem(self): return float(abs(np.dot(self.S("f_stem"), self.S("ax0"))))
    def settle(self,n=1200):
        for _ in range(n): mujoco.mj_step(self.m,self.d)
        self.f_az0=self.f_az(); self.f_stem0=self.f_stem()
        self.q0=self.S("q_fruit").copy()
        mg=FRUIT_MASS*9.81
        self.baseline_ok = max(abs(self.f_az0),abs(self.f_stem0)) <= BASELINE_MAX_x_MG*mg
        return self.f_az0, self.f_stem0, self.baseline_ok, BASELINE_MAX_x_MG*mg
    def measure(self):
        inter,az=self.joint_bends()
        tw=abs(twist_about_axis(quat_mul(self.S("q_fruit"),quat_conj(self.q0)), self.stem_dir))
        F_az=max(0.0,self.f_az()-self.f_az0); F_st=max(0.0,self.f_stem()-self.f_stem0)
        return dict(inter=inter, az_bend=az, twist=tw, f_az=F_az, f_stem=F_st,
                    D_az=az_damage(F_az,az,tw),
                    D_in=[internode_damage(b,tw) for b in inter])
    def verdict(self,z):
        if z["f_stem"]>=STEM_PULL_FORCE: return "STEM_PULL","apple-pedicel joint"
        if z["f_az"]  >=SPUR_BREAK_FORCE: return "SPUR_BREAK","spur"
        if z["D_az"]  >=1.0:              return "SUCCESS","abscission zone"
        j=int(np.argmax(z["D_in"]))
        if z["D_in"][j]>=1.0:             return "STALK_SNAP",f"stalk internode {j+1}/4"
        return None,None


# ---------------------------------------------------------------- stage 4

GRASP_STRENGTH=60.0; FINGER_CLOSE=0.32; FINGER_FORCE=1.5; FINGER_HOLD=1.0
PALM_H=1.47*0.02875
APPROACH_PUSH_MAX=2.5; APPROACH_RISE_MAX=0.0015; APPROACH_COMPRESS_MAX=0.0015
SEAT_CURV_LIMIT=18.0; GRASP_CURV_ABORT=15.0

def weld_here(m, d, name="grasp"):
    """Activate a weld at the CURRENT relative pose.

    MuJoCo stores a weld's relpose in mjModel and, if it is not given, falls back to
    the relative pose at qpos0 -- not at the moment of activation. Switching the
    weld on without rewriting eq_data therefore snaps the fruit back to wherever it
    sat when the model was compiled. Here that was a 397 N yank and a 289 deg stalk.
    """
    eq=m.equality(name); i=eq.id
    b1,b2 = m.eq_obj1id[i], m.eq_obj2id[i]
    p1,q1 = d.xpos[b1].copy(), d.xquat[b1].copy()
    p2,q2 = d.xpos[b2].copy(), d.xquat[b2].copy()
    q1i=np.zeros(4); mujoco.mju_negQuat(q1i,q1)
    dp=np.zeros(3); mujoco.mju_rotVecQuat(dp,p2-p1,q1i)
    dq=np.zeros(4); mujoco.mju_mulQuat(dq,q1i,q2)
    m.eq_data[i,0:3]=0.0          # anchor, in body2 frame
    m.eq_data[i,3:6]=dp
    m.eq_data[i,6:10]=dq
    m.eq_data[i,10]=1.0           # torquescale
    d.eq_active[i]=1

def weld_force(m, d, name="grasp"):
    """Magnitude of the force the grasp weld is carrying, in N.

    PyBullet capped its grasp constraint at GRASP_STRENGTH and let it fail above
    that. MuJoCo equality constraints have no force limit, so the cap has to be
    applied by hand -- which means reading the constraint rows out of efc.
    """
    i=m.equality(name).id
    rows=[k for k in range(d.nefc)
          if d.efc_type[k]==mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[k]==i]
    if not rows: return 0.0
    return float(np.linalg.norm(d.efc_force[rows][:3]))

# ---------------------------------------------------------------- stage 5
APPROACH_FORCE=8.0; WRIST_FORCE_ROLL=10.0; WRIST_FORCE_PULL=120.0
MAX_BEND=np.deg2rad(60); MAX_TWIST=np.deg2rad(75); LAG_MAX=np.deg2rad(10)
BEND_RATE=np.deg2rad(35); TWIST_RATE=np.deg2rad(25); PULL_RATE=0.02
SIM_HZ=240

def axis_angle_quat(axis, ang):
    axis=np.asarray(axis,float); n=np.linalg.norm(axis)
    if n<1e-12: return np.array([1.,0,0,0])
    axis=axis/n
    return np.concatenate([[np.cos(ang/2)], np.sin(ang/2)*axis])

def qrot(q, v):
    out=np.zeros(3); mujoco.mju_rotVecQuat(out, np.asarray(v,float), np.asarray(q,float)); return out

def _rooted_at(m, i, root):
    while i > 0:
        if i == root: return True
        i = int(np.atleast_1d(m.body_parentid[i])[0])
    return False


class WristServo:
    """A force-limited pose servo on the palm, replacing the kinematic mocap base.

    This is not a refinement. PyBullet drove the gripper through a fixed constraint
    with `maxForce`, and the build's own note fixes why that number is load-bearing:

        wrist roll (10 N) < stem pull (18 N) < AZ tension (30 N) < spur break (45 N)

    "The rolling wrist must never be able to out-muscle the weakest link on its own."
    At 25 N it sat above the stem-pull threshold and a moment of poor tracking tore
    the stalk out, so the label recorded the servo rather than the physics. A mocap
    body has no force limit at all - it sits above every rung - and a sweep driven
    that way measures the driver, not the fruit.

    Gains are chosen so the cap does not bind during free motion: it binds only when
    something resists, which is the behaviour being reproduced. The palm's mass and
    the gains themselves are solver detail, not frozen constants.

    **kp is bounded by the integrator, not by tracking.** Critical damping sets
    kd = 2*sqrt(kp*m), and an explicit step is only stable while kd*dt/m stays well
    under one. At kp = 5000 that ratio is 1.08: the damping term overshoots every step
    and the servo self-excites, with the force cap clipping the divergence into a
    sustained limit cycle - half a millimetre of chatter per step through the whole
    approach. The same pattern as the palm's rotational inertia in the slip study, where
    a torque cap was quietly holding a diverging orientation servo together.

    kp = 1000 puts the ratio at 0.48 and the chatter drops fifty-fold. The seven-row
    sweep returns the same outcome class in all seven, and the do-nothing row improves:
    at kp = 5000 the shake was injecting 2.2 degrees of abscission bend into a motion
    that commands none.
    """
    def __init__(self, m, d, kp=1000.0, kr=50.0, torque_arm=0.02875):
        self.m, self.d = m, d
        self.bid = m.body("palm").id
        self.kp, self.kr, self.torque_arm = kp, kr, torque_arm
        self.kd = 2.0*np.sqrt(kp*m.body_mass[self.bid])
        # the palm plus everything rigidly hanging off it - the three fingers
        self.hold_mass = float(sum(m.body_mass[i] for i in range(m.nbody)
                                   if _rooted_at(m, i, self.bid)))
        self.krd = 2.0*np.sqrt(kr*float(np.mean(m.body_inertia[self.bid])))
    def drive(self, pos_t, quat_t, f_cap):
        d, m = self.d, self.m
        b = self.bid
        e = np.asarray(pos_t,float) - d.xpos[b]
        v = d.cvel[b][3:6]
        # Hold your own weight without asking position error to do it. Without this the
        # palm drops at t=0, the spring winds up, the cap clips the WHOLE vector - damping
        # included - and the servo settles into a bang-bang limit cycle: about half a
        # millimetre a step, for the entire approach. A real wrist does not sag two
        # millimetres to carry its own arm, and gravity is known exactly, so it is
        # feed-forward rather than something the loop has to discover.
        w = np.zeros(3); w[2] = -self.m.opt.gravity[2]*self.hold_mass
        f = self.kp*e - self.kd*v + w
        n = np.linalg.norm(f)
        if n > f_cap: f *= f_cap/n
        qi=np.zeros(4); mujoco.mju_negQuat(qi, d.xquat[b])
        qe=np.zeros(4); mujoco.mju_mulQuat(qe, np.asarray(quat_t,float), qi)
        if qe[0] < 0: qe = -qe
        ang=np.zeros(3); mujoco.mju_quat2Vel(ang, qe, 1.0)
        t = self.kr*ang - self.krd*d.cvel[b][0:3]
        # PyBullet's changeConstraint(maxForce=F) limits a fixed constraint's force AND
        # torque by the same number, so the faithful arm is 1.0, not a characteristic
        # radius. The 0.02875 default is kept only because stages 1-5 were run and frozen
        # with it; switching it there changes nothing (verified bit-identical), but it is
        # not something to alter in a frozen deliverable.
        t_cap = f_cap*self.torque_arm
        n = np.linalg.norm(t)
        if n > t_cap: t *= t_cap/n
        d.xfrc_applied[b,0:3] = f
        d.xfrc_applied[b,3:6] = t
        return float(np.linalg.norm(f))

def wrist_state(m, d, az_pivot, ref_pos, ref_quat, stem_dir):
    """Bend and twist the wrist has ACTUALLY reached - the servo error measure.

    Not the stalk's curvature: the chain bows, so the abscission bend lags the wrist
    by design. Clamping the command against the stalk angle would stall the motion.
    """
    b = m.body("palm").id
    v0 = np.asarray(ref_pos,float) - az_pivot
    v1 = d.xpos[b] - az_pivot
    n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
    bend = float(np.arccos(np.clip(float(np.dot(v0/n0, v1/n1)), -1, 1))) if n0>1e-9 and n1>1e-9 else 0.0
    qi=np.zeros(4); mujoco.mju_negQuat(qi, np.asarray(ref_quat,float))
    qr=np.zeros(4); mujoco.mju_mulQuat(qr, d.xquat[b], qi)
    return bend, abs(twist_about_axis(qr, stem_dir))

# ---------------------------------------------------------------- stage 7
FRUIT_INERTIA = np.array([2.15835e-5, 2.15835e-5, 2.43276e-5])
BASELINE_MAX_x_MG = 3.0
SEAT_MAX_x_PALM_H = 1.35
CURV_REST_MAX_DEG = 15.0

def set_segment_mass(m, mass, inertia=None):
    """Segment mass as a runtime parameter, so one scene serves both configurations.

    The frozen json records 2 g; the notebook overrode it to 20 g immediately after
    construction, and the acceptance tests ran on units that never saw the override.
    Rather than keep two scenes, the mass is set here and every result is labelled
    with the value it was produced at.
    """
    for name in ("seg0","seg1","seg2","seg3","seg4"):
        b = m.body(name).id
        m.body_mass[b] = mass
        if inertia is not None:
            m.body_inertia[b] = np.asarray(inertia, float)

def harvest(rig, sv, ref_pos, mode, bend_deg_s=35.0, twist_deg_s=25.0, act_steps=2500):
    """One picking attempt on an already-grasped rig. Returns the same fields the
    PyBullet build's harvest() reported, so the acceptance checks read the same way.

    `pull` moves the palm AWAY from the branch along the stem direction. Driving it
    the other way loads the stalk in compression, which the axial force reading
    cannot tell from tension - it fired STEM_PULL at one mass and SUCCESS at the
    other before the sign was fixed.
    """
    m, d = rig.m, rig.d
    az = d.xpos[m.body("seg4").id].copy()
    B = T = 0.0
    peak_stem = 0.0
    for i in range(act_steps):
        if mode == "pull":
            sv.drive(ref_pos - np.array([0,0,1.])*PULL_RATE*i/SIM_HZ, np.array([1.,0,0,0]),
                     WRIST_FORCE_PULL)
        else:
            wb, wt = wrist_state(m, d, az, ref_pos, np.array([1.,0,0,0]), np.array([0.,0,1.]))
            T = min(T + np.deg2rad(twist_deg_s)/SIM_HZ, MAX_TWIST, wt + LAG_MAX)
            B = min(B + np.deg2rad(bend_deg_s)/SIM_HZ, MAX_BEND,  wb + LAG_MAX)
            qb = axis_angle_quat(np.array([1.,0,0]), B)
            sv.drive(az + qrot(qb, ref_pos - az),
                     quat_mul(qb, quat_mul(axis_angle_quat(np.array([0.,0,1.]), T),
                                           np.array([1.,0,0,0]))), WRIST_FORCE_ROLL)
        mujoco.mj_step(m, d)
        z = rig.measure()
        peak_stem = max(peak_stem, z["f_stem"])
        code, where = rig.verdict(z)
        if code: break
    else:
        code, where = "NO_DETACH", "none"; z = rig.measure()
    inter, _ = rig.joint_bends()
    return dict(code=code, where=where,
                bend_deg=np.degrees(z["az_bend"]), twist_deg=np.degrees(z["twist"]),
                peak_stem_force=peak_stem,
                curv_internode_deg=float(np.degrees(sum(inter))))

# ---------------------------------------------------------------- stage 9
def transfer_state(m0, d0, m1, d1):
    """Carry the whole configuration across to a post-separation model.

    Joints are matched by NAME, never by index: the second model inserts a free joint
    that the first does not have, so index-wise copying silently shifts the whole stalk.

    Whichever free joint is new in m1 is the one the break created, and the body that
    owns it is seeded from its world pose and velocity. That keeps the same code
    working whether the parting is at the abscission layer (the pedicel leaves with the
    fruit) or at the fruit-stalk joint (the fruit leaves bare).
    """
    names0 = {mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m0.njnt)}
    new_free = None
    for j in range(m1.njnt):
        name = mujoco.mj_id2name(m1, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in names0:
            j0 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, name)
            n_q={mujoco.mjtJoint.mjJNT_FREE:7, mujoco.mjtJoint.mjJNT_BALL:4}.get(m1.jnt_type[j],1)
            n_v={mujoco.mjtJoint.mjJNT_FREE:6, mujoco.mjtJoint.mjJNT_BALL:3}.get(m1.jnt_type[j],1)
            a1,a0=m1.jnt_qposadr[j], m0.jnt_qposadr[j0]
            b1,b0=m1.jnt_dofadr[j],  m0.jnt_dofadr[j0]
            d1.qpos[a1:a1+n_q]=d0.qpos[a0:a0+n_q]
            d1.qvel[b1:b1+n_v]=d0.qvel[b0:b0+n_v]
        elif m1.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:
            new_free = j
    if new_free is not None:
        b1 = int(np.atleast_1d(m1.jnt_bodyid[new_free])[0])
        name = mujoco.mj_id2name(m1, mujoco.mjtObj.mjOBJ_BODY, b1)
        b0 = m0.body(name).id
        v=np.zeros(6); mujoco.mj_objectVelocity(m0,d0,mujoco.mjtObj.mjOBJ_BODY,b0,v,0)
        R=d0.xmat[b0].reshape(3,3)
        a=m1.jnt_qposadr[new_free]; b=m1.jnt_dofadr[new_free]
        d1.qpos[a:a+3]=d0.xpos[b0]; d1.qpos[a+3:a+7]=d0.xquat[b0]
        d1.qvel[b:b+3]=v[3:6]; d1.qvel[b+3:b+6]=R.T@v[0:3]
    d1.ctrl[:]=d0.ctrl[:]
    mujoco.mj_forward(m1,d1)


# ---------------------------------------------------------------- shared run pipeline
Q0_  = np.array([1., 0, 0, 0])
STEM_= np.array([0., 0, 1.])
PERP_= np.array([1., 0, 0])

def fingers_on_fruit(rig):
    """Fingers touching the fruit, and total normal force.

    Fingers, not contact points: the PyBullet clip overlaid the point count and
    contradicted the paper's three-finger claim.
    """
    m, d = rig.m, rig.d
    hit=set(); N=0.0
    for i in range(d.ncon):
        g=(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,d.contact[i].geom1),
           mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,d.contact[i].geom2))
        if "apple" not in g: continue
        f=np.zeros(6); mujoco.mj_contactForce(m,d,i,f); N+=abs(f[0])
        for s in g:
            if s.startswith(("glo","gup")): hit.add(s[-1])
    return len(hit), N

def approach_close(rig, sv, weld=True, weld_at="seat", settle=6000, on_step=None):
    """Park, rise until seated, form the grasp, close the cage, let the bow converge.

    `weld_at` decides whether the grasp forms at seating or after the fingers have shut.
    The original forms it after; doing the same here spins the fruit about 40 degrees
    while the cage closes and leaves 24 degrees of it standing. That is not a solver
    artefact - swapping the ellipsoid for a sphere changes nothing - it is the stalk
    having almost no torsional restoring (0.0055 N.m/rad in series) against the small
    azimuthal moment three closing contacts cannot avoid (0.006 N.m). Nothing in the
    outcome depends on it, since the act phase takes its reference afterwards, but the
    clip shows an apple spinning in a still gripper, and the grasp then starts already
    loaded to 4.1 N against a 2.2 N guard.

    Forming the weld at seating fixes both: the spin goes to zero and the baseline drops
    to 1.3 N, inside the guard, with the same outcome. It also matches what the weld is
    for - the assumption that the gripper holds what it has arrived under.

    `on_step(phase)` is called every step so a viewer or a recorder can follow along
    without this being written twice.
    """
    def _grasp():
        weld_here(m, d); m.actuator_forcerange[:] = [[-FINGER_HOLD, FINGER_HOLD]]*3
    m, d = rig.m, rig.d
    def hold(p, f, n, phase):
        for _ in range(n):
            sv.drive(p, Q0_, f); mujoco.mj_step(m, d)
            if on_step: on_step(phase)
    d.ctrl[:]=0.0
    hold([0,0,-0.12], APPROACH_FORCE, 1200, "park")
    tgt=d.body("fruit").xpos[2]-PALM_H; z=-0.12
    while z<tgt:
        z=min(tgt, z+0.0002); hold([0,0,z], APPROACH_FORCE, 20, "approach")
        # Stop on push, not on geometry. APPROACH_PUSH_MAX has been a constant since the
        # start and nothing was reading it: the rise ran to a computed target regardless,
        # and if the cage met the fruit early the servo simply leaned on it. Demand then
        # sits above the 8 N cap, the controller saturates into bang-bang, and the palm
        # chatters half a millimetre a step for the whole approach - visible in a clip
        # sampled at real time, invisible in one sampled every eightieth step.
        if fingers_on_fruit(rig)[1] > APPROACH_PUSH_MAX:
            break
    hold([0,0,z], APPROACH_FORCE, 600, "seated")
    if weld and weld_at == "seat":
        _grasp()
    for k in range(1,2401):
        d.ctrl[:]=min(FINGER_CLOSE, FINGER_CLOSE*k/1200)
        sv.drive([0,0,z], Q0_, APPROACH_FORCE); mujoco.mj_step(m,d)
        if on_step: on_step("close")
    if weld and weld_at == "close":
        _grasp()
    hold([0,0,z], APPROACH_FORCE, settle, "settle")
    rig.settle(0)
    return np.array([0,0,z])

def act(rig, sv, ref, bend=45, twist=15, steps=2500, on_step=None):
    """The picking motion. The palm orbits the MEASURED abscission point, so no stretch
    is injected into a chain that has no axial give."""
    m, d = rig.m, rig.d
    az = d.xpos[m.body("seg4").id].copy()
    B=T=0.0
    for i in range(steps):
        wb, wt = wrist_state(m, d, az, ref, Q0_, STEM_)
        T=min(T+np.deg2rad(twist)/SIM_HZ, MAX_TWIST, wt+LAG_MAX)
        B=min(B+np.deg2rad(bend)/SIM_HZ,  MAX_BEND,  wb+LAG_MAX)
        qb=axis_angle_quat(PERP_, B)
        sv.drive(az+qrot(qb, ref-az),
                 quat_mul(qb, quat_mul(axis_angle_quat(STEM_, T), Q0_)), WRIST_FORCE_ROLL)
        mujoco.mj_step(m,d)
        if on_step: on_step("act")
        code, where = rig.verdict(rig.measure())
        if code: return code, where, i
    return "NO_DETACH", "none", steps

def retreat(rig2, sv2, steps=800, distance=0.09, on_step=None):
    """Withdraw 90 mm away from the branch, as the original does."""
    pal = rig2.d.xpos[rig2.m.body("palm").id].copy()
    qp  = rig2.d.xquat[rig2.m.body("palm").id].copy()
    for k in range(1, steps+1):
        sv2.drive(pal - STEM_*(distance*k/steps), qp, WRIST_FORCE_PULL)
        mujoco.mj_step(rig2.m, rig2.d)
        if on_step: on_step("retreat")

SEPARATION_SCENE = {"SUCCESS":    "harvest_scene_az.xml",     # pedicel leaves with the fruit
                    "STEM_PULL":  "harvest_scene_stem.xml",   # fruit leaves bare
                    "STALK_SNAP": "harvest_scene_stem.xml",   # nearest available approximation
                    "SPUR_BREAK": "harvest_scene_az.xml"}

def detach_into(rig, code="SUCCESS", weld=True):
    """Switch to the model that matches how the chain actually parted.

    The outcome decides the geometry. A SUCCESS separates at the abscission layer, so
    the whole pedicel travels with the fruit and only the spur is left behind; a
    STEM_PULL tears the stalk out of the fruit and leaves it on the tree. Using one
    scene for both shows the wrong failure.
    """
    xml_detached = SEPARATION_SCENE.get(code, "harvest_scene_az.xml")
    r2 = Rig.__new__(Rig)
    r2.m = mujoco.MjModel.from_xml_path(_scene(xml_detached))
    r2.d = mujoco.MjData(r2.m); r2.stem_dir = STEM_.copy()
    transfer_state(rig.m, rig.d, r2.m, r2.d)
    if weld: weld_here(r2.m, r2.d)
    return r2, WristServo(r2.m, r2.d, torque_arm=1.0)

def make_rig(xml="harvest_scene.xml"):
    r = Rig.__new__(Rig)
    r.m = mujoco.MjModel.from_xml_path(_scene(xml)); r.d = mujoco.MjData(r.m)
    r.stem_dir = STEM_.copy()
    mujoco.mj_forward(r.m, r.d)
    r.settle(0)          # record a baseline immediately, so measure() is callable from
                         # step one; approach_close re-records it once the grasp settles
    return r, WristServo(r.m, r.d, torque_arm=1.0)

def render_setup(m, w=800, h=600, alpha=0.65, dist_mult=3.4, lookat_z=-0.055):
    """Camera, contact-point display, and the clip-only fruit transparency.

    The scene keeps the fruit opaque: 14.95 of the stalk's 22.25 mm sits inside the
    fruit silhouette, so transparency is what makes the chain visible, and it is a
    rendering choice that must never reach the physics model.
    """
    rend = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, lookat_z]
    cam.distance = 0.0575*dist_mult; cam.azimuth = 40.0; cam.elevation = -5.0
    m.vis.global_.fovy = 50.0
    opt = mujoco.MjvOption(); opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    a = m.geom("apple").id
    m.geom_rgba[a] = [*m.geom_rgba[a][:3], alpha]
    return rend, cam, opt
