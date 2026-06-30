"""RL configuration for Z01 AMP locomotion task."""

import os
from dataclasses import dataclass, field
from typing import List

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from src.tasks.amp_loco.config.z01.env_cfgs import Z01_LOCOMOTION_JOINT_NAMES

_MOTION_DATA_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
  "src", "assets", "motions", "Z01", "amp",
)

Z01_NUM_DOFS = len(Z01_LOCOMOTION_JOINT_NAMES)


@dataclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Extended runner config with AMP-specific parameters."""
  amp_reward_coef: float = 0.1
  amp_motion_files: str = ""
  amp_num_preload_transitions: int = 200000
  amp_task_reward_lerp: float = 0.75
  amp_discr_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])
  min_normalized_std: List[float] = field(default_factory=lambda: [0.05] * Z01_NUM_DOFS)
  max_normalized_std: List[float] = field(default_factory=lambda: [2.0] * Z01_NUM_DOFS)
  amp_body_names: tuple = ()
  amp_anchor_name: str = ""


def _z01_amp_base_runner_cfg(
  *,
  experiment_name: str,
  init_std: float,
  learning_rate: float,
  entropy_coef: float,
  max_normalized_std: float,
  max_iterations: int,
) -> RslRlAmpRunnerCfg:
  return RslRlAmpRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": init_std,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=entropy_coef,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=learning_rate,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      class_name="AMPPPO",
    ),
    clip_actions=True,
    experiment_name=experiment_name,
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=max_iterations,
    amp_reward_coef=0.1,
    amp_motion_files=os.path.normpath(_MOTION_DATA_DIR),
    amp_num_preload_transitions=200000,
    amp_task_reward_lerp=0.75,
    amp_discr_hidden_dims=[1024, 512, 256],
    min_normalized_std=[0.05] * Z01_NUM_DOFS,
    max_normalized_std=[max_normalized_std] * Z01_NUM_DOFS,
    amp_body_names=(
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
    ),
    amp_anchor_name="waist_pitch_link",
  )


def z01_amp_ppo_warmup_runner_cfg() -> RslRlAmpRunnerCfg:
  """Conservative RL config for flat warmup (phase 1)."""
  cfg = _z01_amp_base_runner_cfg(
    experiment_name="z01_amp_warmup",
    init_std=0.3,
    learning_rate=3.0e-4,
    entropy_coef=0.001,
    max_normalized_std=0.8,
    max_iterations=10000,
  )
  # Give AMP style reward more weight during warmup (task reward still dominant).
  cfg.amp_task_reward_lerp = 0.5
  return cfg


def z01_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """RL config for flat/rough training after warmup (phase 2/3)."""
  return _z01_amp_base_runner_cfg(
    experiment_name="z01_amp_locomotion",
    init_std=0.5,
    learning_rate=5.0e-4,
    entropy_coef=0.003,
    max_normalized_std=2.0,
    max_iterations=100001,
  )
