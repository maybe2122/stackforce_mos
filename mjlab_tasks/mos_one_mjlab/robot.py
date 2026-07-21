"""mos2026_2 robot definition for mjlab.

Reuses the existing MJCF (deploy/mujoco/assets/mos2026_2.xml, produced by
tools/asset/usd_to_mjcf.py, closed-chain <connect> already configured) and
adapts it programmatically for mjlab:
  - strips the embedded floor / light (mjlab's terrain entity provides both),
  - strips the embedded <position> actuators (mjlab adds its own from
    BuiltinPositionActuatorCfg, aligned with the Isaac training setup),
  - adds an IMU site + the sensor suite mjlab's velocity task expects
    (imu_ang_vel / imu_lin_vel / imu_lin_acc / imu_upvector / root_angmom),
  - adds per-foot sites (FL/FR/RL/RR) at the shank tips, computed from the
    shank mesh's most distal vertex cluster (see doc/mjlab_integration.md).

Actuator constants: GO-M8010-6, N=6.33 (see 传统步态/代码/dynamics.py).
PD gains + effort limit mirror the Isaac env (kp=25, kd=0.5, effort=12) so a
policy trained here stays comparable with the Isaac baseline.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_REPO_ROOT = Path(__file__).resolve().parents[2]
MOS_XML: Path = _REPO_ROOT / "deploy" / "mujoco" / "assets" / "mos2026_2.xml"
assert MOS_XML.exists(), f"MJCF not found: {MOS_XML}"

# The 12 actuated joints (exact names — fl knee joint is `fl_shank_link`, the
# other three are `*_shank_link_a`; regexes would also catch the passive
# `*_shank_link_b` loop joints, so we enumerate explicitly).
ACTUATED_JOINTS = (
  "fl_hip", "fr_hip", "rl_hip", "rr_hip",
  "fl_thigh", "fr_thigh", "rl_thigh", "rr_thigh",
  "fl_shank_link", "fr_shank_link_a", "rl_shank_link_a", "rr_shank_link_a",
)

# Foot tip in the shank body frame: most distal vertex cluster of the shank
# collision mesh (computed once via mesh analysis; all four legs agree to <1mm).
FOOT_SITES: dict[str, tuple[str, tuple[float, float, float]]] = {
  "FL": ("fl_shank", (0.1637, -0.0136, -0.1433)),
  "FR": ("fr_shank", (0.1636, 0.0115, -0.1434)),
  "RL": ("rl_shank", (0.1636, -0.0136, -0.1433)),
  "RR": ("rr_shank", (0.1637, 0.0115, -0.1433)),
}

FOOT_GEOM_NAMES = tuple(f"{leg}_shank_geom" for leg in ("fl", "fr", "rl", "rr"))


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(MOS_XML))

  # mjlab provides terrain + lighting; the embedded floor would collide with it.
  for geom in list(spec.geoms):
    if geom.name == "floor":
      spec.delete(geom)
  for light in list(spec.lights):
    spec.delete(light)

  # mjlab builds actuators from the EntityCfg below; drop the embedded ones.
  for act in list(spec.actuators):
    spec.delete(act)

  # IMU site at the base origin + the sensors mjlab's velocity task reads.
  base = spec.body("base")
  base.add_site(name="imu", pos=[0.0, 0.0, 0.0], group=5)
  spec.add_sensor(
    name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
  )
  spec.add_sensor(
    name="imu_lin_vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
  )
  spec.add_sensor(
    name="imu_lin_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
  )
  upvec = spec.add_sensor(
    name="imu_upvector", type=mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
    objtype=mujoco.mjtObj.mjOBJ_BODY, objname="world",
  )
  upvec.reftype = mujoco.mjtObj.mjOBJ_SITE
  upvec.refname = "imu"
  spec.add_sensor(
    name="root_angmom", type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY, objname="base",
  )

  # Per-foot sites at the shank tips (contact bookkeeping + clearance rewards).
  for site_name, (body_name, pos) in FOOT_SITES.items():
    spec.body(body_name).add_site(
      name=site_name, pos=list(pos), size=[0.015, 0.015, 0.015], group=5,
    )

  # Closed-chain stability: the <connect> equalities ship with solref
  # timeconst=0.002 (tuned for play_mujoco's 1 kHz stepping). Soft-constraint
  # stability requires timeconst >= 2*dt; the mjlab task runs dt=0.002, so
  # clamp the loop solref up to 0.004 (still stiff: <1 mm anchor breathing).
  for eq in spec.equalities:
    if eq.type == mujoco.mjtEq.mjEQ_CONNECT and eq.solref[0] < 0.004:
      eq.solref[0] = 0.004

  return spec


##
# Actuators — GO-M8010-6 @ N=6.33 (传统步态/代码/dynamics.py), gains/effort
# aligned with the Isaac env (stiffness 25 / damping 0.5 / effort_limit_sim 12).
##

ROTOR_INERTIA = 0.000111842  # 8010-series rotor (same family as Go1), kg·m²
GEAR_RATIO = 6.33
ARMATURE = ROTOR_INERTIA * GEAR_RATIO**2  # ≈0.00448 kg·m² reflected

MOS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=ACTUATED_JOINTS,
  stiffness=25.0,
  damping=0.5,
  effort_limit=12.0,
  armature=ARMATURE,
)

MOS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(MOS_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.9,
)

##
# Initial state — mirrors the Isaac env's init_state (hips widened to ±0.15,
# thighs/shank-links at 0). Base starts slightly above the settled height.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.32),
  joint_pos={
    "fl_hip": 0.15,
    "fr_hip": 0.15,
    "rl_hip": -0.15,
    "rr_hip": -0.15,
    ".*_thigh": 0.0,
    ".*_shank.*": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision — the converted MJCF collides via the `*_geom` mesh geoms
# (contype=2/conaffinity=1 from the XML defaults). Feet (shank meshes) get
# friction + condim 3; everything else stays frictionless contact (condim 1).
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_geom",),
  # Original XML semantics: contype=2/conaffinity=1 → robot-robot pairs never
  # collide (2&1=0 both ways) but robot-terrain (contype 1) does. The closed
  # -chain links overlap geometrically; CollisionCfg's default contype=1/
  # conaffinity=1 would enable self-collision and blow the loops up at spawn.
  contype=2,
  conaffinity=1,
  condim={"^(fl|fr|rl|rr)_shank_geom$": 3, ".*_geom": 1},
  priority={"^(fl|fr|rl|rr)_shank_geom$": 1},
  friction={"^(fl|fr|rl|rr)_shank_geom$": (1.0,)},
)


def get_mos_robot_cfg() -> EntityCfg:
  """Fresh EntityCfg per call (mjlab configs are mutated by env cfgs)."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=MOS_ARTICULATION,
  )


# Residual action scale — mirrors the Isaac env (hips 0.5, thighs/knees 0.8145).
MOS_ACTION_SCALE: dict[str, float] = {
  ".*_hip": 0.5,
  ".*_thigh": 0.8145,
  "fl_shank_link": 0.8145,
  ".*_shank_link_a": 0.8145,
}


if __name__ == "__main__":
  # Standalone sanity check: compile the adapted spec.
  spec = get_spec()
  model = spec.compile()
  print(f"OK: nq={model.nq} nu={model.nu} nsensor={model.nsensor} nsite={model.nsite}")
