from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

import torch

from mjlab.sensor import ContactSensor

from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_error_magnitude
from mjlab.managers.manager_term_config import RewardTermCfg
from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: Optional[list[str]]
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: Optional[list[str]] = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: Optional[list[str]] = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: Optional[list[str]] = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: Optional[list[str]] = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Cost that returns the number of self-collisions detected by a sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)



###box task

def _get_box_task(env, cfg=None):
    if not hasattr(env, "_box_task_module"):
        assert cfg is not None, "first call needs cfg to build TaskRewardModule"
        env._box_task_module = TaskRewardModule(cfg, env)
    return env._box_task_module

def catching_point(env):
    mod = _get_box_task(env)
    return mod.catching_point()

def box_target_position_reward(env, target_xy=(0.3, 0.0)):
    mod = _get_box_task(env)
    return mod.box_target_position_reward(env, target_xy=target_xy)

def box_smooth_motion_reward(env):
    mod = _get_box_task(env)
    return mod.box_smooth_motion_reward(env)

def box_grasp_force_reward(env, F_target: float = 20.0, sigma: float = 5.0):
    mod = _get_box_task(env)
    return mod.box_grasp_force_reward(env, F_target=F_target, sigma=sigma)

class TaskRewardModule:
    """合法的任务奖励模块，保存状态并提供多个可独立调用的奖励函数"""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        # 任务状态
        self.has_caught = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._has_caught_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.has_placed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.update_time = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        # 初始化共享缓存
        self._dist_mean = torch.zeros(env.num_envs, device=env.device)
        self._box_pos = torch.zeros(env.num_envs,3, device=env.device)
        self._box_quat = torch.zeros(env.num_envs,4, device=env.device)
        self.grasp_threshold = 25.0 #抓取的力大于该值则算处于抓取状态
        self.box_force_threshold = 12.0 #box抓取判断

        #把实例化挂载
        env._box_task_module = self

    def __call__(self, env: ManagerBasedRlEnv, command_name: str):
        ##无奈之举，由于类的实例化自动完成，无法加入额外参数
        ##任务奖励，抓取放下完成
        command = cast(MotionCommand, env.command_manager.get_term(command_name))
        time_step = command.time_steps #B
        reset_envs = time_step == 0
        if torch.any(reset_envs):
            self.reset(reset_envs.nonzero(as_tuple=True)[0])
        self._update_state(env)
        reward=self.has_placed.float()*0.1 + self.has_caught.float()*0.2
        return reward

    def _update_state(self,env: ManagerBasedRlEnv):
        # 计算抓取点的距离
        robot = env.scene["robot"]
        box = env.scene["box"]
        box_sites = ["grasp_left", "grasp_right"]
        box_ids = [box.site_names.index(s) for s in box_sites]
        hand_sites = ["left_hand", "right_hand"]
        hand_ids = [robot.site_names.index(s) for s in hand_sites]
        box_pos = box.data.site_pos_w[:, box_ids, :]
        hand_pos = robot.data.site_pos_w[:, hand_ids, :]
        dist = torch.norm(hand_pos - box_pos, dim=-1)
        self._dist_mean = dist.mean(dim=-1)
        # 更新箱子高度
        self._box_pos = box.data.geom_pos_w[:, 0, :]
        self._box_quat = box.data.geom_quat_w[:, 0, :]
        # 更新手上力传感器，由于自碰撞存在，期望不会发生意外碰撞
        left_box_grasp = env.scene["robot/hand_left_touch"].data  # [N_envs, 3]
        right_box_grasp = env.scene["robot/hand_right_touch"].data  # [N_envs, 3]
        left_fmag = torch.norm(left_box_grasp, dim=-1)
        right_fmag = torch.norm(right_box_grasp, dim=-1)
        box_force = env.scene["box/grasp_left_force"].data #左右结果都一样，记录的是geom的力
        box_force_fmag = torch.norm(box_force, dim=-1)
        # print("left_fmag",left_fmag,"right_fmag",right_fmag,"box_force_fmag",box_force_fmag)
        # 更新抓取状态
        self.has_caught = (left_fmag > self.grasp_threshold) & (right_fmag > self.grasp_threshold) & (box_force_fmag > self.box_force_threshold)
        self.has_placed |= (~self.has_caught) & self._has_caught_prev
        #has_place与自己或运算，一旦进入hasplace就一直hasplace
        self._has_caught_prev = self.has_caught.clone()

    def catching_point(self)->torch.Tensor:
        """
        当手靠近箱子时（距离小于 d_thresh），给予平滑奖励；
        当 has_placed == True 时，奖励停止。
        """
        # 参数设定
        d_thresh = 0.2
        sigma = 0.05
        # 取缓存中的距离与状态
        dist = self._dist_mean  # [num_envs]
        placed = self.has_placed
        # 计算高斯型接近奖励
        near_mask = (dist < d_thresh) & (~placed)
        # print("near_mask",near_mask)
        reward = torch.exp(-0.5 * (dist / sigma) ** 2) * near_mask.float()
        return reward

    def box_target_position_reward(self, env: ManagerBasedRlEnv,target_xy: tuple[float, float] = (0.3, 0.0)) -> torch.Tensor:
        """
        奖励箱子水平位置靠近目标 (x, y)，
        仅在 has_caught == True（被拿起时）才开始计算。
        """
        target_x, target_y = target_xy
        box_xy = self._box_pos[:, :2]  # 只取x, y
        dist_xy = torch.norm(box_xy - torch.tensor([target_x, target_y], device=env.device), dim=-1)

        caught_mask = self.has_caught & (~self.has_placed)
        sigma = 0.1  # 平滑半径
        reward = torch.exp(-0.5 * (dist_xy / sigma) ** 2) * caught_mask.float()
        return reward

    def box_smooth_motion_reward(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        """
        奖励箱子平滑移动：速度与角速度越小奖励越高。
        """
        box = env.scene["box"]
        lin_vel = box.data.root_com_lin_vel_w  # [B, 3]
        ang_vel = box.data.root_com_ang_vel_w  # [B, 3]

        lin_speed = torch.norm(lin_vel, dim=-1)
        ang_speed = torch.norm(ang_vel, dim=-1)

        # 指数惩罚型：速度越小，值越接近1
        lin_rew = torch.exp(-2.0 * lin_speed ** 2)
        ang_rew = torch.exp(-1.0 * ang_speed ** 2)

        # 平滑奖励在抓取或搬运阶段激活（未放下）
        active_mask = self.has_caught & (~self.has_placed)
        reward = (0.7 * lin_rew + 0.3 * ang_rew) * active_mask.float()

        return reward

    def box_grasp_force_reward(self, env: ManagerBasedRlEnv, F_target: float = 20.0,
                               sigma: float = 5.0) -> torch.Tensor:
        """
        当 has_caught == True 时，鼓励抓取力接近目标值 (默认 20 N)。
        """
        left_force = env.scene["robot/hand_left_touch"].data
        right_force = env.scene["robot/hand_right_touch"].data

        F_left = torch.norm(left_force, dim=-1)
        F_right = torch.norm(right_force, dim=-1)
        F_mean = 0.5 * (F_left + F_right)

        # 高斯型接近奖励：力越接近目标值越好
        reward = torch.exp(-0.5 * ((F_mean - F_target) / sigma) ** 2)

        # 仅在抓取中生效
        active_mask = self.has_caught & (~self.has_placed)
        reward = reward * active_mask.float()
        return reward

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            self.has_caught.zero_()
            self._has_caught_prev.zero_()
            self.has_placed.zero_()
        else:
            self.has_caught[env_ids] = False
            self._has_caught_prev[env_ids] = False
            self.has_placed[env_ids] = False

# def catching_point(env: ManagerBasedRlEnv) -> torch.Tensor:
#     robot=env.scene.entities["robot"]
#     box=env.scene.entities["box"]
#     box_site_name = ["grasp_left", "grasp_right"]
#     hand_site_id = [box.site_name_to_idx(name) for name in box_site_name]
#     hand_site_name = ["left_hand", "right_hand"]
#     hand_site_id = [robot.site_name_to_idx(name) for name in hand_site_name]
#     hand_site = robot.data.site_pos_w[:,hand_site_id,:]#B,2,3
#     box_site = box.data.site_pos_w[:,box_site_id,:]#B,2,3
#     dist = torch.norm(hand_site_pos - box_site_pos, dim=-1)  # [B, 2]
#     dist_mean = dist.mean(dim=-1)  # [B]
#     return 0

# def contacting