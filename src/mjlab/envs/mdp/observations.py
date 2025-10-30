"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_BOX_CFG = SceneEntityCfg("box")

##
# Root state.
##
def whole_body_angmom(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """
  整机角动量，形状 (N, 3)。来源：EntityData.whole_body_subtreeangmom
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.whole_body_subtreeangmom  # (N, 3)



def base_lin_vel(
  env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b


def base_ang_vel(
  env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b


def projected_gravity(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b


##
# Joint state.
##


def joint_pos_rel(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_pos[:, jnt_ids] - default_joint_pos[:, jnt_ids]


def joint_vel_rel(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, jnt_ids] - default_joint_vel[:, jnt_ids]


##
# Actions.
##


def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_action

##
# Sin wave
##


def sin_wave(env: ManagerBasedEnv,
) -> torch.Tensor:
    sin_wave=torch.sin()
    cos_wave=torch.cos()
    return 0

##
# Commands.
##


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command


##
# Sensors.
##


def builtin_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Get observation from a built-in sensor by name."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return sensor.data

##
# box.
##
def box_pos_w(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = _BOX_CFG,
) -> torch.Tensor:
    """
    世界系下盒子位置
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w

def box_quat_w(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = _BOX_CFG,
) -> torch.Tensor:
    """世界坐标下的 box 姿态（四元数）"""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_quat_w

def box_lin_vel_w(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = _BOX_CFG,
) -> torch.Tensor:
    """世界坐标下的 box 线速度"""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_w

def box_ang_vel_b(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = _BOX_CFG,
) -> torch.Tensor:
    """世界坐标下的 box 角速度"""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_b

