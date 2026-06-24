"""Z01 robot constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

Z01_XML: Path = SRC_PATH / "assets" / "robots" / "Z01" / "xml" / "Z01.xml"
assert Z01_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, Z01_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(Z01_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0


def _make_actuator(
  armature: float,
  effort_limit: float,
  target_names_expr: tuple[str, ...],
) -> BuiltinPositionActuatorCfg:
  stiffness = armature * NATURAL_FREQ**2
  damping = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=stiffness,
    damping=damping,
    effort_limit=effort_limit,
    armature=armature,
  )


Z01_ACTUATOR_HIP_KNEE = _make_actuator(
  armature=0.08,
  effort_limit=330.0,
  target_names_expr=("left_hip_pitch", "right_hip_pitch", "left_knee_pitch", "right_knee_pitch"),
)
Z01_ACTUATOR_HIP_YR = _make_actuator(
  armature=0.03,
  effort_limit=70.0,
  target_names_expr=("left_hip_roll", "right_hip_roll", "left_hip_yaw", "right_hip_yaw"),
)
Z01_ACTUATOR_ANKLE = _make_actuator(
  armature=0.02,
  effort_limit=75.0,
  target_names_expr=(
    "left_ankle_pitch",
    "right_ankle_pitch",
    "left_ankle_roll",
    "right_ankle_roll",
  ),
)
Z01_ACTUATOR_WAIST = _make_actuator(
  armature=0.04,
  effort_limit=102.0,
  target_names_expr=("waist_yaw", "waist_pitch"),
)
Z01_ACTUATOR_SHOULDER = _make_actuator(
  armature=0.02,
  effort_limit=66.0,
  target_names_expr=(
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_shoulder_roll",
    "right_shoulder_roll",
  ),
)
Z01_ACTUATOR_ELBOW = _make_actuator(
  armature=0.01,
  effort_limit=34.0,
  target_names_expr=(
    "left_elbow_yaw",
    "right_elbow_yaw",
    "left_elbow_pitch",
    "right_elbow_pitch",
  ),
)
Z01_ACTUATOR_WRIST = _make_actuator(
  armature=0.005,
  effort_limit=11.0,
  target_names_expr=("left_wrist_yaw", "right_wrist_yaw"),
)
Z01_ACTUATOR_HEAD = _make_actuator(
  armature=0.002,
  effort_limit=11.0,
  target_names_expr=("head_yaw", "head_pitch", "head_roll"),
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.96),
  joint_pos={
    "left_hip_pitch": 0.03,
    "left_hip_roll": 0.16,
    "left_hip_yaw": -0.19,
    "left_knee_pitch": 0.0,
    "left_ankle_pitch": -0.18,
    "left_ankle_roll": -0.22,
    "right_hip_pitch": 0.0,
    "right_hip_roll": 0.13,
    "right_hip_yaw": -0.36,
    "right_knee_pitch": 0.09,
    "right_ankle_pitch": -0.20,
    "right_ankle_roll": -0.13,
    "waist_yaw": -0.43,
    "waist_pitch": -0.22,
    "left_shoulder_pitch": 0.09,
    "left_shoulder_roll": 0.31,
    "left_elbow_yaw": -0.02,
    "left_elbow_pitch": 0.0,
    "left_wrist_yaw": -0.02,
    "right_shoulder_pitch": -0.04,
    "right_shoulder_roll": -0.58,
    "right_elbow_yaw": 0.01,
    "right_elbow_pitch": 0.0,
    "right_wrist_yaw": 0.01,
    "head_yaw": 0.0,
    "head_pitch": 0.0,
    "head_roll": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r".*ankle_roll_link_collision$": 3, ".*_collision": 1},
  priority={r".*ankle_roll_link_collision$": 1},
  friction={r".*ankle_roll_link_collision$": (0.6,)},
)

##
# Final config.
##

Z01_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    Z01_ACTUATOR_HIP_KNEE,
    Z01_ACTUATOR_HIP_YR,
    Z01_ACTUATOR_ANKLE,
    Z01_ACTUATOR_WAIST,
    Z01_ACTUATOR_SHOULDER,
    Z01_ACTUATOR_ELBOW,
    Z01_ACTUATOR_WRIST,
    Z01_ACTUATOR_HEAD,
  ),
  soft_joint_pos_limit_factor=0.9,
)

Z01_JOINT_NAMES = (
  "left_hip_pitch",
  "left_hip_roll",
  "left_hip_yaw",
  "left_knee_pitch",
  "left_ankle_pitch",
  "left_ankle_roll",
  "right_hip_pitch",
  "right_hip_roll",
  "right_hip_yaw",
  "right_knee_pitch",
  "right_ankle_pitch",
  "right_ankle_roll",
  "waist_yaw",
  "waist_pitch",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_elbow_yaw",
  "left_elbow_pitch",
  "left_wrist_yaw",
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_elbow_yaw",
  "right_elbow_pitch",
  "right_wrist_yaw",
  "head_yaw",
  "head_pitch",
  "head_roll",
)


def get_z01_robot_cfg() -> EntityCfg:
  """Get a fresh Z01 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=Z01_ARTICULATION,
  )


Z01_ACTION_SCALE: dict[str, float] = {}
for a in Z01_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    Z01_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_z01_robot_cfg())
  viewer.launch(robot.spec.compile())
