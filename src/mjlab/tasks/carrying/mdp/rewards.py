from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_lin_vel_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """线速度跟踪奖励"""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  desired = torch.zeros_like(actual)
  desired[:, :2] = command[:, :2]
  lin_vel_error = torch.sum(torch.square(desired - actual), dim=1)
  return torch.exp(-lin_vel_error / std**2)

def track_ang_vel_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """跟踪yaw的奖励"""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  desired = torch.zeros_like(actual)
  desired[:, 2] = command[:, 2]
  ang_vel_error = torch.sum(torch.square(desired - actual), dim=1)
  return torch.exp(-ang_vel_error / std**2)

def torso_ang_vel_xy(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚xy轴的旋转，pitch和roll"""
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel_b = asset.data.root_link_ang_vel_b
    # print(f"ang_vel_b = {ang_vel_b}")
    cost = torch.sum(torch.square(ang_vel_b[:, :2]), dim=1)
    return cost

def joint_vel_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """关节速度惩罚"""
  asset: Entity = env.scene[asset_cfg.name]
  joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  cost = torch.sum(torch.square(joint_vel), dim=1)
  return cost

def feet_distance_exp(
    env: ManagerBasedRlEnv,
    left_geom_id: int,
    right_geom_id: int,
    target_y_offset: float = 0.14,
    scale: float = 5.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """左右脚不要偏移太远"""
  asset: Entity = env.scene[asset_cfg.name]
  #世界系下
  feet_pos_w = asset.data.geom_pos_w[:, [left_geom_id, right_geom_id], :]
  root_pos_w = asset.data.root_link_pos_w
  root_quat_w = asset.data.root_link_quat_w
  #转到机器人质心系
  feet_rel_w = feet_pos_w - root_pos_w.unsqueeze(1)
  num_feet = feet_rel_w.shape[1]
  root_quat_expand = root_quat_w.unsqueeze(1).repeat(1, num_feet, 1)
  feet_pos_b = quat_apply_inverse(
      root_quat_expand.reshape(-1, 4),
      feet_rel_w.reshape(-1, 3)
  ).view(root_quat_w.shape[0], num_feet, 3)

  left_y = feet_pos_b[:, 0, 1]
  right_y = feet_pos_b[:, 1, 1]
  # print("left",left_y,"right", right_y)

  left_foot_dist = torch.abs(left_y - target_y_offset)
  right_foot_dist = torch.abs(right_y + target_y_offset)
  reward = torch.exp(-scale * left_foot_dist) + torch.exp(-scale * right_foot_dist)
  return reward

def feet_ori_xy(
    env: ManagerBasedRlEnv,
    body_ids: list[int],
    scale: float = 5.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward feet staying flat in pitch/roll (xy orientation)."""
  asset: Entity = env.scene[asset_cfg.name]
  gravity_vec = asset.data.gravity_vec_w
  foot_quats = asset.data.body_link_quat_w[:, body_ids, :]  # 脚的姿态
  reward = 0.0
  for i in range(len(body_ids)):
    projected_gravity = quat_apply_inverse(foot_quats[:, i, :], gravity_vec)
    # XY方向分量反映了倾斜程度
    tilt_xy = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
    reward += torch.exp(-scale * tilt_xy)
  return reward

def feet_ori_z(
    env: ManagerBasedRlEnv,
    body_ids: list[int],
    scale: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward foot yaw alignment with base orientation."""
  asset: Entity = env.scene[asset_cfg.name]

  gravity_y = torch.tensor([[0.0, 1.0, 0.0]], device=env.device).repeat((env.num_envs, 1))
  base_quat = asset.data.root_link_quat_w
  base_y = quat_apply_inverse(base_quat, gravity_y)

  foot_quats = asset.data.body_link_quat_w[:, body_ids, :]
  rewards = torch.zeros(env.num_envs, device=env.device)

  for i in range(len(body_ids)):
    foot_y = quat_apply_inverse(foot_quats[:, i, :], gravity_y)
    feet_angle = torch.atan(-base_y[:, 0] / base_y[:, 1]) - torch.atan(
      -foot_y[:, 0] / foot_y[:, 1]
    )
    rewards += torch.exp(-scale * torch.abs(feet_angle))

  return rewards

def angmom_reward(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    axis_weight: tuple[float, float, float] = (0.0, 0.0, 1.0),
    kernel: str = "exp",   # "exp" | "l2" | "l2sq"
    scale: float = 10.0,   # 仅对 "exp" 有效
) -> torch.Tensor:
    """
    奖励整机质心角动量接近 0
    """
    asset: Entity = env.scene[asset_cfg.name]
    H = asset.data.whole_body_subtreeangmom #N,3
    w = torch.tensor(axis_weight, device=env.device, dtype=H.dtype).view(1, 3)
    H_w = H * w
    mag = torch.linalg.norm(H_w, dim=-1)  # (N,)
    if kernel == "exp":
        return torch.exp(-scale * mag)
    elif kernel == "l2":
        return -mag
    elif kernel == "l2sq":
        return -torch.square(mag)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

class feet_swing_height:
  """Reward correct swing (air) height of feet.

  When a foot is in the air, reward its z-position being close to a target height.
  No reward (or penalty) is given when the foot is in contact with the ground.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    # ---- 参数读取 ----
    self.asset_name = cfg.params["asset_name"]
    self.sensor_names = cfg.params["sensor_names"]
    self.geom_ids = cfg.params["geom_ids"]
    self.target_height = cfg.params.get("target_height", 0.35)
    self.contact_threshold = cfg.params.get("contact_threshold", 1.0)
    self.command_name = cfg.params.get("command_name", None)
    self.command_threshold = cfg.params.get("command_threshold", 0.05)
    self.command_scale_type = cfg.params.get("command_scale_type", "smooth")
    self.command_scale_width = cfg.params.get("command_scale_width", 0.2)
    self.num_feet = len(self.sensor_names)
    asset: Entity = env.scene[self.asset_name]
    for sensor_name in self.sensor_names:
      if sensor_name not in asset.sensor_names:
        raise ValueError(f"Sensor '{sensor_name}' not found in asset '{self.asset_name}'")

    # 初始化张量状态
    self._zeros = torch.zeros(env.num_envs, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """在每次 step() 中计算奖励"""
    asset: Entity = env.scene[self.asset_name]

    contact_list, height_list = [], []
    for i, sensor_name in enumerate(self.sensor_names):
      # 读取接触力传感器数据
      sensor_data = asset.data.sensor_data[sensor_name]
      in_contact = sensor_data[:, 0] > self.contact_threshold
      contact_list.append(in_contact)
      # 读取足端高度
      foot_z = asset.data.geom_pos_w[:, self.geom_ids[i], 2]
      height_list.append(foot_z)

    # 拼接每只脚的接触状态与高度
    in_contact = torch.stack(contact_list, dim=1)  # (envs, num_feet)
    in_air = ~in_contact
    feet_z = torch.stack(height_list, dim=1)
    # 对未接触地面的脚计算偏差
    pos_error = torch.square(feet_z - self.target_height) * in_air
    reward = torch.mean(pos_error, dim=1)

    # 如果配置了命令调制（随速度命令变化）
    if self.command_name is not None:
      command = env.command_manager.get_command(self.command_name)
      assert command is not None
      command_norm = torch.norm(command[:, :2], dim=1)
      if self.command_scale_type == "smooth":
        scale = 0.5 * (
          1.0 + torch.tanh((command_norm - self.command_threshold) / self.command_scale_width)
        )
        reward *= scale
      else:
        reward *= (command_norm > self.command_threshold)
    return torch.abs(reward)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    """无状态可重置，但保持接口一致性"""
    return

class feet_air_time:
  """
  在“落地瞬间”结算一次 (air_time - target)^2 的代价（按步频离散化为每步奖励）。
  配合负权重使用（weight < 0），即可最小化误差平方。
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.asset_name = cfg.params["asset_name"]
    self.sensor_names = cfg.params["sensor_names"]
    self.num_feet = len(self.sensor_names)

    self.target_time = cfg.params.get("target_time", 0.40)  # 目标腾空时长（秒）
    self.contact_threshold = cfg.params.get("contact_threshold", 1.0)

    # 可选命令门控（保持和你现有term一致）
    self.command_name = cfg.params.get("command_name", None)
    self.command_threshold = cfg.params.get("command_threshold", 0.05)
    self.command_scale_type = cfg.params.get("command_scale_type", "smooth")
    self.command_scale_width = cfg.params.get("command_scale_width", 0.2)

    # 计时器
    self.current_air_time = torch.zeros(env.num_envs, self.num_feet, device=env.device)
    self.last_air_time = torch.zeros(env.num_envs, self.num_feet, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    asset: Entity = env.scene[self.asset_name]

    # 接触判定（沿用你 feet_land_time 的范式：用前三轴力范数与阈值比较）
    in_contact_list = []
    for sensor_name in self.sensor_names:
      sensor_data = asset.data.sensor_data[sensor_name]
      in_contact = sensor_data[:, 0] > self.contact_threshold
      in_contact_list.append(in_contact)
    in_contact = torch.stack(in_contact_list, dim=1)
    in_air = ~in_contact
    # “首次接触”=落地事件（上一拍在空中，当前拍接触）
    first_contact = (self.current_air_time > 0) & in_contact
    # print("first_contact",first_contact)
    # 记录本次腾空时长
    self.last_air_time = torch.where(first_contact, self.current_air_time, self.last_air_time)
    # print("last_air_time",self.last_air_time)

    # 更新时间器
    self.current_air_time = torch.where(
      in_contact,
      torch.zeros_like(self.current_air_time),            # 接触中清零
      self.current_air_time + env.step_dt,               # 空中累加
    )
    error = (self.last_air_time - self.target_time) ** 2
    reward = torch.sum(error * first_contact.float(), dim=1) / env.step_dt

    # print("reward",reward,"error",error)

    if self.command_name is not None:
      command = env.command_manager.get_command(self.command_name)
      assert command is not None
      command_norm = torch.norm(command[:, :2], dim=1)
      if self.command_scale_type == "smooth":
        scale = 0.5 * (1.0 + torch.tanh((command_norm - self.command_threshold) / self.command_scale_width))
        reward *= scale
      else:
        reward *= (command_norm > self.command_threshold)

    return reward  # 注意：配合负权重使用（最小化误差平方）

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    self.current_air_time[env_ids] = 0.0
    self.last_air_time[env_ids] = 0.0
    # print("reset!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# class feet_stance_time_target:
#   """
#   在“抬脚瞬间”结算一次 (contact_time - target)^2 的代价（按步频离散化为每步奖励）。
#   配合负权重使用（weight < 0）。
#   """
#
#   def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
#     self.asset_name = cfg.params["asset_name"]
#     self.sensor_names = cfg.params["sensor_names"]
#     self.num_feet = len(self.sensor_names)
#
#     self.target_time = cfg.params.get("target_time", 0.3)  # 目标支撑时长（秒）
#     self.contact_threshold = cfg.params.get("contact_threshold", 1.0)
#
#     self.command_name = cfg.params.get("command_name", None)
#     self.command_threshold = cfg.params.get("command_threshold", 0.05)
#     self.command_scale_type = cfg.params.get("command_scale_type", "smooth")
#     self.command_scale_width = cfg.params.get("command_scale_width", 0.2)
#
#     self.current_contact_time = torch.zeros(env.num_envs, self.num_feet, device=env.device)
#     self.last_contact_time = torch.zeros(env.num_envs, self.num_feet, device=env.device)
#
#   def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
#     asset: Entity = env.scene[self.asset_name]
#
#     in_contact_list = []
#     for sensor_name in self.sensor_names:
#       sensor_data = asset.data.sensor_data[sensor_name]
#       in_contact = sensor_data[:, 0] > self.contact_threshold
#       in_contact_list.append(in_contact)
#     in_contact = torch.stack(in_contact_list, dim=1)
#     in_air = ~in_contact
#
#     # “首次离地”=抬脚事件（上一拍在地面，当前拍在空中）
#     first_lift = (self.current_contact_time > 0) & in_air
#
#     # 记录本次着地时长
#     self.last_contact_time = torch.where(first_lift, self.current_contact_time, self.last_contact_time)
#     # 更新时间器
#     self.current_contact_time = torch.where(
#       in_contact,
#       self.current_contact_time + env.step_dt,       # 接触中累加
#       torch.zeros_like(self.current_contact_time),   # 空中清零
#     )
#
#     # 抬脚瞬间结算误差平方
#     error = (self.last_contact_time - self.target_time) ** 2
#     reward = torch.sum(error * first_lift.float(), dim=1) / env.step_dt
#
#     # 命令门控（可选）
#     if self.command_name is not None:
#       command = env.command_manager.get_command(self.command_name)
#       assert command is not None
#       command_norm = torch.norm(command[:, :2], dim=1)
#       if self.command_scale_type == "smooth":
#         scale = 0.5 * (1.0 + torch.tanh((command_norm - self.command_threshold) / self.command_scale_width))
#         reward *= scale
#       else:
#         reward *= (command_norm > self.command_threshold)
#
#     return reward  # 注意：配合负权重使用
#
#   def reset(self, env_ids: torch.Tensor | slice | None = None):
#     if env_ids is None:
#       env_ids = slice(None)
#     self.current_contact_time[env_ids] = 0.0
#     self.last_contact_time[env_ids] = 0.0
#     # print("reset!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
class feet_stance_time_target:
    """
    连续版本：在接触期间持续奖励支撑时间接近 target_time。
    配合正权重使用（weight > 0）。
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.asset_name = cfg.params["asset_name"]
        self.sensor_names = cfg.params["sensor_names"]
        self.num_feet = len(self.sensor_names)

        self.target_time = cfg.params.get("target_time", 0.40)
        self.contact_threshold = cfg.params.get("contact_threshold", 1.0)

        self.command_name = cfg.params.get("command_name", None)
        self.command_threshold = cfg.params.get("command_threshold", 0.05)
        self.command_scale_type = cfg.params.get("command_scale_type", "smooth")
        self.command_scale_width = cfg.params.get("command_scale_width", 0.2)

        self.current_contact_time = torch.zeros(env.num_envs, self.num_feet, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
        asset: Entity = env.scene[self.asset_name]

        in_contact_list = []
        for sensor_name in self.sensor_names:
            sensor_data = asset.data.sensor_data[sensor_name]
            in_contact = sensor_data[:, 0] > self.contact_threshold
            in_contact_list.append(in_contact)
        in_contact = torch.stack(in_contact_list, dim=1)

        self.current_contact_time = torch.where(
            in_contact,
            self.current_contact_time + env.step_dt,
            torch.zeros_like(self.current_contact_time),
        )

        penalty = -(self.current_contact_time - self.target_time).clamp(min=0.0)
        reward = torch.sum(penalty, dim=1)

        if self.command_name is not None:
            command = env.command_manager.get_command(self.command_name)
            assert command is not None
            command_norm = torch.norm(command[:, :2], dim=1)
            if self.command_scale_type == "smooth":
                scale = 0.5 * (
                    1.0 + torch.tanh((command_norm - self.command_threshold) / self.command_scale_width)
                )
                reward *= scale
            else:
                reward *= (command_norm > self.command_threshold)

        return reward  # 配合负权重使用（越接近目标越好）

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.current_contact_time[env_ids] = 0.0

def foot_clearance_reward(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  tanh_mult: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  foot_z_target_error = torch.square(
    asset.data.geom_pos_w[:, asset_cfg.geom_ids, 2] - target_height
  )
  foot_velocity_tanh = torch.tanh(
    tanh_mult * torch.norm(asset.data.geom_lin_vel_w[:, asset_cfg.geom_ids, :2], dim=2)
  )
  reward = foot_z_target_error * foot_velocity_tanh
  return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_names: list[str],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  contact_list = []
  for sensor_name in sensor_names:
    sensor_data = asset.data.sensor_data[sensor_name]
    foot_contact = sensor_data[:, 0] > 0
    contact_list.append(foot_contact)
  contacts = torch.stack(contact_list, dim=1)
  geom_vel = asset.data.geom_lin_vel_w[:, asset_cfg.geom_ids, :2]
  return torch.sum(geom_vel.norm(dim=-1) * contacts, dim=1)
