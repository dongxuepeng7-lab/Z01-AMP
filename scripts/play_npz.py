#!/usr/bin/env python3
"""Replay and validate AMP motion NPZ files in MuJoCo.

Modes
-----
- joint_air: Fix the root in the air and replay joint positions only.
  Useful for checking joint order, sign, and range without balance effects.
- full: Replay root pose + joints from the NPZ (same as training reset data).
- check: Run full replay without a viewer and report FK error vs stored body states.

Examples
--------
  # Suspend robot and replay joints in a viewer
  python scripts/play_npz.py \\
    --npz src/assets/motions/Z01/amp/WalkandRun/walk1_subject1.npz \\
    --robot z01 --mode joint_air

  # Full trajectory replay
  python scripts/play_npz.py \\
    --npz src/assets/motions/Z01/amp/WalkandRun/walk1_subject1.npz \\
    --robot z01 --mode full --realtime

  # Numeric validation only (no window)
  python scripts/play_npz.py \\
    --npz src/assets/motions/Z01/amp/WalkandRun/walk1_subject1.npz \\
    --robot z01 --mode check
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import mujoco.viewer as mj_viewer
import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.terrains import TerrainEntityCfg

from src.assets.robots import Z01_JOINT_NAMES, get_z01_robot_cfg

# Only used when --robot g1. Z01 uses Z01_JOINT_NAMES from z01_constants.py.
G1_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]


def _z01_flat_scene_cfg() -> SceneCfg:
  return SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": get_z01_robot_cfg()},
    num_envs=1,
    extent=2.0,
  )


def _get_robot_scene_and_joints(robot: Literal["g1", "z01"]) -> tuple[SceneCfg, list[str]]:
  if robot == "g1":
    return unitree_g1_flat_tracking_env_cfg().scene, G1_JOINT_NAMES
  if robot == "z01":
    return _z01_flat_scene_cfg(), list(Z01_JOINT_NAMES)
  raise ValueError(f"Unsupported robot: {robot}")


@dataclass
class NpzMotion:
  path: Path
  device: torch.device

  def __post_init__(self) -> None:
    data = np.load(self.path)
    required = ("fps", "joint_pos", "joint_vel")
    missing = [k for k in required if k not in data.files]
    if missing:
      raise KeyError(f"{self.path}: missing keys {missing}")

    self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=self.device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=self.device)
    self.num_frames, self.num_joints = self.joint_pos.shape

    self.body_pos_w = None
    self.body_quat_w = None
    self.body_lin_vel_w = None
    self.body_ang_vel_w = None
    if "body_pos_w" in data.files:
      self.body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=self.device)
      self.body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=self.device)
      self.body_lin_vel_w = torch.tensor(
        data["body_lin_vel_w"], dtype=torch.float32, device=self.device
      )
      self.body_ang_vel_w = torch.tensor(
        data["body_ang_vel_w"], dtype=torch.float32, device=self.device
      )
      self.num_bodies = self.body_pos_w.shape[1]
    else:
      self.num_bodies = 0

  @property
  def duration_s(self) -> float:
    return self.num_frames / self.fps


def _build_sim(robot: Literal["g1", "z01"], fps: float, device: str) -> tuple[Simulation, Scene, list[str]]:
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / fps

  scene_cfg, joint_names = _get_robot_scene_and_joints(robot)
  scene = Scene(scene_cfg, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return sim, scene, joint_names


def _validate_joint_dim(motion: NpzMotion, robot_joint_count: int, npz_path: Path) -> None:
  if motion.num_joints != robot_joint_count:
    raise ValueError(
      f"{npz_path}: joint_pos has {motion.num_joints} DOFs but robot expects "
      f"{robot_joint_count}. Joint order may be wrong."
    )


def _apply_joint_state(
  robot: Entity,
  joint_ids: list[int] | torch.Tensor,
  motion: NpzMotion,
  frame_idx: int,
) -> None:
  joint_pos = robot.data.default_joint_pos.clone()
  joint_vel = robot.data.default_joint_vel.clone()
  joint_pos[:, joint_ids] = motion.joint_pos[frame_idx].unsqueeze(0)
  joint_vel[:, joint_ids] = motion.joint_vel[frame_idx].unsqueeze(0)
  robot.write_joint_state_to_sim(joint_pos, joint_vel)


def _apply_full_frame(
  robot: Entity,
  scene: Scene,
  motion: NpzMotion,
  frame_idx: int,
  root_body_index: int,
) -> None:
  if motion.body_pos_w is None or motion.body_quat_w is None:
    raise ValueError("NPZ has no body_pos_w/body_quat_w; use mode=joint_air instead.")

  root_pos = motion.body_pos_w[frame_idx, root_body_index].unsqueeze(0).clone()
  root_pos[:, :2] += scene.env_origins[:, :2]
  root_quat = motion.body_quat_w[frame_idx, root_body_index].unsqueeze(0)

  root_pose = torch.cat([root_pos, root_quat], dim=-1)
  robot.write_root_link_pose_to_sim(root_pose)

  if motion.body_lin_vel_w is not None and motion.body_ang_vel_w is not None:
    lin_vel = motion.body_lin_vel_w[frame_idx, root_body_index].unsqueeze(0)
    ang_vel = motion.body_ang_vel_w[frame_idx, root_body_index].unsqueeze(0)
    root_vel = torch.cat([lin_vel, ang_vel], dim=-1)
    robot.write_root_link_velocity_to_sim(root_vel)


def _apply_joint_air_frame(
  robot: Entity,
  scene: Scene,
  suspend_height: float,
) -> None:
  root_states = robot.data.default_root_state.clone()
  root_states[:, 0] = scene.env_origins[:, 0]
  root_states[:, 1] = scene.env_origins[:, 1]
  root_states[:, 2] = suspend_height
  root_states[:, 3:7] = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0]], device=root_states.device, dtype=root_states.dtype
  )
  root_states[:, 7:] = 0.0
  robot.write_root_state_to_sim(root_states)


def _sync_viewer(sim: Simulation, window_viewer: mujoco.viewer.Handle) -> None:
  if sim.mj_model.nq > 0:
    sim.mj_data.qpos[:] = sim.data.qpos[0].cpu().numpy()
    sim.mj_data.qvel[:] = sim.data.qvel[0].cpu().numpy()
  if sim.mj_model.nmocap > 0:
    sim.mj_data.mocap_pos[:] = sim.data.mocap_pos[0].cpu().numpy()
    sim.mj_data.mocap_quat[:] = sim.data.mocap_quat[0].cpu().numpy()
  sim.mj_data.xfrc_applied[:] = sim.data.xfrc_applied[0].cpu().numpy()
  mujoco.mj_forward(sim.mj_model, sim.mj_data)
  window_viewer.sync()


def _print_motion_summary(motion: NpzMotion, robot: str) -> None:
  print("=" * 72)
  print(f"NPZ: {motion.path}")
  print(f"robot={robot}, frames={motion.num_frames}, joints={motion.num_joints}, fps={motion.fps:.1f}")
  print(f"duration={motion.duration_s:.2f}s, bodies={motion.num_bodies}")
  if motion.body_pos_w is not None:
    root_z = motion.body_pos_w[:, 0, 2].cpu().numpy()
    print(
      "root height (body 0): "
      f"min={root_z.min():.3f}, max={root_z.max():.3f}, mean={root_z.mean():.3f} m"
    )
  joint_range = (motion.joint_pos.max(dim=0).values - motion.joint_pos.min(dim=0).values).cpu().numpy()
  print(
    "joint range: "
    f"min={joint_range.min():.3f}, max={joint_range.max():.3f}, mean={joint_range.mean():.3f} rad"
  )
  print("=" * 72)


def run_fk_check(
  motion: NpzMotion,
  robot: Entity,
  scene: Scene,
  sim: Simulation,
  joint_ids: list[int] | torch.Tensor,
  root_body_index: int,
  frame_start: int,
  frame_end: int,
) -> bool:
  """Return True if FK errors are within tolerance."""
  if motion.body_pos_w is None:
    print("[check] body_pos_w missing — skipping FK comparison.")
    return True

  sim_bodies = robot.data.body_link_pos_w.shape[1]
  npz_bodies = motion.body_pos_w.shape[1]
  compare_bodies = min(sim_bodies, npz_bodies)
  if sim_bodies != npz_bodies:
    print(
      f"[check] body count mismatch: sim={sim_bodies}, npz={npz_bodies}. "
      f"Comparing first {compare_bodies} bodies."
    )

  pos_errors: list[float] = []
  root_pos_errors: list[float] = []
  root_height_errors: list[float] = []

  for frame_idx in range(frame_start, frame_end):
    _apply_full_frame(robot, scene, motion, frame_idx, root_body_index)
    _apply_joint_state(robot, joint_ids, motion, frame_idx)
    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    sim_pos = robot.data.body_link_pos_w[0, :compare_bodies]
    ref_pos = motion.body_pos_w[frame_idx, :compare_bodies]
    err = torch.norm(sim_pos - ref_pos, dim=-1)
    pos_errors.append(float(err.max()))

    root_err = torch.norm(sim_pos[root_body_index] - ref_pos[root_body_index]).item()
    root_pos_errors.append(root_err)
    root_height_errors.append(abs(sim_pos[root_body_index, 2].item() - ref_pos[root_body_index, 2].item()))

  print("\n[check] FK error vs stored body_pos_w")
  print(f"  all bodies  max err: {max(pos_errors):.6f} m, mean max/frame: {np.mean(pos_errors):.6f} m")
  print(f"  root body   max err: {max(root_pos_errors):.6f} m, mean: {np.mean(root_pos_errors):.6f} m")
  print(f"  root height max err: {max(root_height_errors):.6f} m, mean: {np.mean(root_height_errors):.6f} m")

  ok = max(pos_errors) < 0.05
  if ok:
    print("[check] PASS — NPZ is consistent with the robot model.")
  else:
    print(
      "[check] FAIL — large FK error. Likely causes: wrong robot model, "
      "joint order mismatch, or quaternion layout."
    )
  return ok


def play_motion(
  npz: Path,
  robot: Literal["g1", "z01"] = "z01",
  mode: Literal["joint_air", "full", "check"] = "joint_air",
  device: str = "cuda:0",
  suspend_height: float = 1.2,
  root_body_index: int = 0,
  frame_start: int = 0,
  frame_end: int | None = None,
  loop: bool = True,
  realtime: bool = True,
  realtime_scale: float = 1.0,
) -> None:
  if not npz.exists():
    raise FileNotFoundError(npz)

  motion = NpzMotion(path=npz, device=torch.device(device))
  _print_motion_summary(motion, robot)

  sim, scene, joint_names = _build_sim(robot, motion.fps, device)
  robot_entity: Entity = scene["robot"]
  joint_ids = robot_entity.find_joints(joint_names, preserve_order=True)[0]
  _validate_joint_dim(motion, len(joint_names), npz)

  print(f"Robot model: {robot} ({len(joint_names)} joints)")
  print(f"Joint order: {', '.join(joint_names[:4])} ... {', '.join(joint_names[-2:])}")

  end = frame_end if frame_end is not None else motion.num_frames
  end = min(end, motion.num_frames)
  start = max(0, frame_start)
  if start >= end:
    raise ValueError(f"Invalid frame range: [{start}, {end})")

  scene.reset()

  if mode == "check":
    run_fk_check(motion, robot_entity, scene, sim, joint_ids, root_body_index, start, end)
    return

  print(f"\nPlaying mode={mode}, frames=[{start}, {end}), loop={loop}, fps={motion.fps:.1f}")
  if mode == "joint_air":
    print(f"Root fixed at z={suspend_height:.2f} m (joint-only replay).")
  else:
    print(f"Full replay using root body index={root_body_index}.")

  with mj_viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
    while viewer.is_running():
      wall_start = time.perf_counter()
      for local_i, frame_idx in enumerate(range(start, end)):
        if not viewer.is_running():
          break

        if mode == "joint_air":
          _apply_joint_air_frame(robot_entity, scene, suspend_height)
        else:
          _apply_full_frame(robot_entity, scene, motion, frame_idx, root_body_index)

        _apply_joint_state(robot_entity, joint_ids, motion, frame_idx)
        sim.forward()
        scene.update(sim.mj_model.opt.timestep)
        _sync_viewer(sim, viewer)

        if realtime:
          target_t = (local_i + 1) / motion.fps / max(realtime_scale, 1e-6)
          sleep_s = target_t - (time.perf_counter() - wall_start)
          if sleep_s > 0:
            time.sleep(sleep_s)

      if not loop:
        # Keep last frame visible until the window is closed.
        while viewer.is_running():
          _sync_viewer(sim, viewer)
          time.sleep(0.05)
        break


def main(
  npz: Path,
  robot: Literal["g1", "z01"] = "z01",
  mode: Literal["joint_air", "full", "check"] = "joint_air",
  device: str = "cuda:0",
  suspend_height: float = 1.2,
  root_body_index: int = 0,
  frame_start: int = 0,
  frame_end: int | None = None,
  loop: bool = True,
  realtime: bool = True,
  realtime_scale: float = 1.0,
) -> None:
  """Replay / validate an AMP motion NPZ file."""
  play_motion(
    npz=npz,
    robot=robot,
    mode=mode,
    device=device,
    suspend_height=suspend_height,
    root_body_index=root_body_index,
    frame_start=frame_start,
    frame_end=frame_end,
    loop=loop,
    realtime=realtime,
    realtime_scale=realtime_scale,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
