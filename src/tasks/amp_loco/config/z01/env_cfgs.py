"""Z01 AMP Locomotion environment configurations."""

import os

from src.assets.robots import (
  Z01_ACTION_SCALE,
  Z01_JOINT_NAMES,
  get_z01_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg

# Locomotion joints only — head is held at default by PD, not controlled by policy.
Z01_LOCOMOTION_JOINT_NAMES: tuple[str, ...] = tuple(
  name for name in Z01_JOINT_NAMES if not name.startswith("head_")
)
Z01_LOCOMOTION_ACTION_SCALE: dict[str, float] = {
  name: Z01_ACTION_SCALE[name] for name in Z01_LOCOMOTION_JOINT_NAMES
}


def _resolve_z01_motion_dirs() -> tuple[str, str | None]:
  """Resolve WalkandRun / Recovery dirs; recovery is optional."""
  _motion_base = os.path.abspath(
    os.path.join(
      os.path.dirname(__file__), "..", "..", "..", "..", "assets", "motions", "Z01", "amp"
    )
  )
  motion_dir = os.path.join(_motion_base, "WalkandRun")
  recovery_dir = os.path.join(_motion_base, "Recovery")
  if not os.path.isdir(motion_dir):
    motion_dir = _motion_base
  if not os.path.isdir(recovery_dir):
    recovery_dir = None
  return motion_dir, recovery_dir


def _apply_z01_stability_overrides(
  cfg: ManagerBasedRlEnvCfg,
  *,
  warmup: bool = False,
) -> None:
  """Apply Z01-specific stability tweaks."""
  if warmup:
    cfg.events.pop("push_robot", None)
    # Warmup: no delayed reset — avoids explosive height-recovery reward hacking.
    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0
    cfg.events["init_motion_loader"].params["max_delay_steps"] = 0
    # Only sample standing/walking frames (exclude fallen poses from mocap).
    cfg.events["init_motion_loader"].params["min_root_height"] = 0.75
    cfg.events["reset_from_motion"].params["min_root_height"] = 0.75
    cfg.rewards["is_terminated"].weight = -50.0
    cfg.rewards["self_collisions"].weight = -0.15
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.rewards["joint_pos_limits"].weight = -5.0
    cfg.rewards["track_root_height"].params["delay_env_rew_ratio"] = 0.0
    cfg.terminations["bad_base_height"].params["minimum_height"] = 0.65

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.debug_vis = False
    twist_cmd.ranges.lin_vel_x = (0.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
  else:
    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.2
    cfg.events["init_motion_loader"].params["max_delay_steps"] = 150
    cfg.rewards["self_collisions"].weight = -0.2
    cfg.terminations["bad_base_height"].params["minimum_height"] = 0.68


def z01_amp_rough_env_cfg(play: bool = False, warmup: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Z01 rough terrain velocity configuration."""
  cfg = make_amp_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 128
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_z01_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  site_names = ("left_ankle_site", "right_ankle_site")
  geom_names = ("left_ankle_roll_link_collision", "right_ankle_roll_link_collision")
  body_names = (
    "base_link",
    "left_hip_roll_link",
    "left_knee_pitch_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_pitch_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_pitch_link",
    "right_wrist_yaw_link",
  )
  anchor_name = "waist_pitch_link"
  root_name = "base_link"

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = Z01_LOCOMOTION_JOINT_NAMES
  joint_pos_action.scale = Z01_LOCOMOTION_ACTION_SCALE

  locomotion_joint_cfg = SceneEntityCfg("robot", joint_names=Z01_LOCOMOTION_JOINT_NAMES)
  for group in ("actor", "critic"):
    cfg.observations[group].terms["joint_pos"].params = {"asset_cfg": locomotion_joint_cfg}
    cfg.observations[group].terms["joint_vel"].params = {"asset_cfg": locomotion_joint_cfg}

  cfg.viewer.body_name = anchor_name

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.25

  # Z01 is heavier (~59 kg); use conservative command ranges during training.
  twist_cmd.ranges.lin_vel_x = (-0.5, 1.5)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = (anchor_name,)

  motion_dir, recovery_dir = _resolve_z01_motion_dirs()
  cfg.events["init_motion_loader"].params["motion_dir"] = motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = recovery_dir
  cfg.events["reset_from_motion"].params["motion_dir"] = motion_dir
  cfg.events["reset_from_motion"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=Z01_LOCOMOTION_JOINT_NAMES
  )

  cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["track_root_height"].weight = 1.0
  cfg.rewards["track_root_height"].params["std"] = 0.25
  cfg.rewards["foot_slip"].weight = -0.35
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.4,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 15.0},
  )
  cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (root_name,)
  cfg.rewards["joint_acc_l2"].weight = -3.5e-7
  cfg.rewards["action_rate_l2"].weight = -0.015
  cfg.rewards["soft_landing"] = RewardTermCfg(
    func=mdp.soft_landing,
    weight=-1e-3,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )

  cfg.observations["critic"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_pos_b"].params["body_cfg"].body_names = body_names
  cfg.observations["critic"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_pos_b"].params["body_cfg"].body_names = body_names
  cfg.observations["amp"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ori_b"].params["body_cfg"].body_names = body_names
  cfg.observations["amp"].terms["body_lin_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_lin_vel_b"].params["body_cfg"].body_names = body_names
  cfg.observations["amp"].terms["body_ang_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ang_vel_b"].params["body_cfg"].body_names = body_names

  cfg.terminations["bad_base_height"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={"minimum_height": 0.72},
  )

  _apply_z01_stability_overrides(cfg, warmup=warmup)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 1.0

  return cfg


def z01_amp_flat_env_cfg(play: bool = False, warmup: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Z01 flat terrain velocity configuration."""
  cfg = z01_amp_rough_env_cfg(play=play, warmup=warmup)

  cfg.sim.njmax = 640
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    # Wider ranges for manual evaluation after training.
    twist_cmd.ranges.lin_vel_x = (-1.0, 2.0)
    twist_cmd.ranges.lin_vel_y = (-0.8, 0.8)
    twist_cmd.ranges.ang_vel_z = (-1.57, 1.57)

  return cfg


def z01_amp_flat_warmup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat terrain with conservative warmup overrides (phase 1)."""
  return z01_amp_flat_env_cfg(play=play, warmup=True)
