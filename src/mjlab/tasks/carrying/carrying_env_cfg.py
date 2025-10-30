"""Velocity tracking task configuration.

This module defines the base configuration for velocity tracking tasks.
Robot-specific configurations are located in the config/ directory.
"""

import math
from dataclasses import dataclass, field
from mjlab.envs.mdp.terminations import nan_detection
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import CurriculumTermCfg as CurrTerm
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import term
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.nan_guard import NanGuardCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from mjlab.asset_zoo.robots.phybot_mini.phybot_mini_constants import phybot_ROBOT_CFG, Carrying_Box_CFG

##
# Scene.
##

SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=5,
  ),
  num_envs=1,
  extent=2.0,
  entities={
        "robot": phybot_ROBOT_CFG,
        "box": Carrying_Box_CFG,
    },
)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  asset_name="robot",
  body_name="",  # Override in robot cfg.
  distance=3.0,
  elevation=-5.0,
  azimuth=90.0,
)

##
# MDP.
##


@dataclass
class ActionCfg:
  joint_pos: mdp.JointPositionActionCfg = term(
    mdp.JointPositionActionCfg,
    asset_name="robot",
    actuator_names=[".*"],
    scale=0.5,
    use_default_offset=True,
  )


@dataclass
class CommandsCfg:
  twist: mdp.UniformVelocityCommandCfg = term(
    mdp.UniformVelocityCommandCfg,
    asset_name="robot",
    resampling_time_range=(3.0, 8.0),
    rel_standing_envs=0.1,
    rel_heading_envs=1.0,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    ranges=mdp.UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.5),
      lin_vel_y=(-0.5, 0.5),
      ang_vel_z=(-1.0, 1.0),
      heading=(-math.pi, math.pi),
    ),
  )


@dataclass
class ObservationCfg:
  @dataclass
  class PolicyCfg(ObsGroup):
    # base_lin_vel: ObsTerm = term(
    #   ObsTerm,
    #   func=mdp.base_lin_vel,
    #   noise=Unoise(n_min=-0.1, n_max=0.1),
    # )

    base_ang_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    )
    base_lin_acc: ObsTerm = term(
      ObsTerm,
      func=mdp.base_lin_acc_imu,
      params={"sensor_name": "imu_acc",},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    )
    projected_gravity: ObsTerm = term(
      ObsTerm,
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    joint_pos: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    )

    actions: ObsTerm = term(ObsTerm, func=mdp.last_action)

    command: ObsTerm = term(
      ObsTerm, func=mdp.generated_commands, params={"command_name": "twist"}
    )

    def __post_init__(self):
      self.enable_corruption = True

  @dataclass
  class PrivilegedCfg(PolicyCfg):

    base_lin_vel: ObsTerm = term(ObsTerm, func=mdp.base_lin_vel)

    feet_height: ObsTerm = term(
      ObsTerm, func=mdp.feet_height_w, params={"geom_ids": [16, 34]},
    )

    box_pos_w: ObsTerm = term(
        ObsTerm,
        func=mdp.box_pos_w,
    )
    box_quat_w: ObsTerm = term(
        ObsTerm,
        func=mdp.box_quat_w,
    )
    box_lin_vel_w: ObsTerm = term(
        ObsTerm,
        func=mdp.box_lin_vel_w,
    )
    box_ang_vel_w: ObsTerm = term(
        ObsTerm,
        func=mdp.box_ang_vel_b,
    )
    def __post_init__(self):
      super().__post_init__()
      self.enable_corruption = False

  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: PrivilegedCfg = field(default_factory=PrivilegedCfg)


@dataclass
class EventCfg:

  reset_base: EventTerm = term(
    EventTerm,
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5), "yaw": (-3.14, 3.14)},
      "velocity_range": {},
    },
  )

  reset_box: EventTerm = term(
      EventTerm,
      func=mdp.reset_entity_to_default,
      mode="reset",
      params={
          "robot_name": "robot",
          "box_name": "box",  # 基于 robot 的位置
          "offset": (0, 0.15, 0.42),  # 相对偏移：x,y,z 可调，约为“肩膀位置”
          # "offset": (0, 0.5, -0.5),
          "rand_range": (0.05, 0.02, 0.01),
          "rand_rot_deg": 5.0,
  },
  )

  reset_robot_joints: EventTerm = term(
    EventTerm,
    func=mdp.reset_joints_by_scale,
    mode="reset",
    params={
      "position_range": (1.0, 1.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    },
  )

  push_robot: EventTerm | None = term(
    EventTerm,
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
  )

  ####域随机化#####
  randomize_box_mass: EventTerm = term(
      EventTerm,
      func=mdp.randomize_field,
      mode="reset",
      params={
          "field": "body_mass",
          "ranges": (0.8, 1.2),  # ±20%
          "distribution": "uniform",
          "operation": "scale",
          "asset_cfg": SceneEntityCfg("box"),
      }
  )

  # randomize_box_size: EventTerm = term(
  #     EventTerm,
  #     func=mdp.randomize_field,
  #     mode="reset",
  #     params={
  #         "field": "geom_size",
  #         "ranges": (0.5, 1.5),  # 对三个轴统一比例缩放
  #         "distribution": "log_uniform",  # 尺寸常用 log-uniform 更自然
  #         "operation": "scale",
  #         "asset_cfg": SceneEntityCfg("box"),
  #     },
  # )

  foot_friction: EventTerm = term(
    EventTerm,
    mode="startup",
    func=mdp.randomize_field,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=[]),
      "operation": "abs",
      "field": "geom_friction",
      "ranges": (0.3, 1.2),
    },
  )

@dataclass
class RewardCfg:
  track_lin_vel_exp: RewardTerm = term(
    RewardTerm,
    func=mdp.track_lin_vel_exp,
    weight=10.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  )

  track_ang_vel_exp: RewardTerm = term(
    RewardTerm,
    func=mdp.track_ang_vel_exp,
    weight=6.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  )

  pose: RewardTerm = term(
    RewardTerm,
    func=mdp.posture,
    weight=3.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
      "std": [],
    },
  )
  is_alive: RewardTerm = term(RewardTerm, func=mdp.is_alive, weight=1.0,)

  action_rate: RewardTerm = term(RewardTerm, func=mdp.action_rate_l2, weight=-0.05,)

  dof_pos_limits: RewardTerm = term(RewardTerm, func=mdp.joint_pos_limits, weight=-5.0)

  action_rate_l2: RewardTerm = term(RewardTerm, func=mdp.action_rate_l2, weight=-0.1)

  torso_ang_vel_xy: RewardTerm = term(RewardTerm, func=mdp.torso_ang_vel_xy, weight=-4.0, )

  torso_orientation: RewardTerm = term(RewardTerm, func=mdp.flat_orientation_l2, weight=-10.0,)

  dof_acc: RewardTerm = term(RewardTerm, func=mdp.joint_acc_l2, weight=-2.5e-7,)

  joint_vel_l2: RewardTerm = term(RewardTerm, func=mdp.joint_vel_l2, weight=-0.01,)

  energy: RewardTerm = term(RewardTerm, func=mdp.electrical_power_cost, weight=-0.001,)

  angmom: RewardTerm = term(RewardTerm,func=mdp.angmom_reward,weight=5.0,)

  # stumble: RewardTerm = term(
  #     RewardTerm,
  #     func=mdp.stumble_penalty,
  #     weight=-1.0,
  #     params={"sensor_names": ["left_foot_ground_contact", "right_foot_ground_contact"],},
  # )

  feet_ori_xy: RewardTerm = term(
      RewardTerm,
      func=mdp.feet_ori_xy,
      weight=1.0,
      params={"body_ids": [6, 12], "scale": 5.0},
  )

  feet_ori_z: RewardTerm = term(
      RewardTerm,
      func=mdp.feet_ori_z,
      weight=1.0,
      params={"body_ids": [6, 12], "scale": 2.0},
  )

  feet_distance: RewardTerm = term(
      RewardTerm,
      func=mdp.feet_distance_exp,
      weight=1.0,
      params={
          "left_geom_id": 16,
          "right_geom_id": 34,
          "target_y_offset": 0.14,
          "scale": 5.0,
      },
  )

  feet_swing_height: RewardTerm = term(
      RewardTerm,
      func=mdp.feet_swing_height,
      weight=-50.0,
      params={
          "asset_name": "robot",
          "target_height": 0.2,
          "contact_threshold": 1.0,
          "sensor_names": ["left_foot_ground_contact", "right_foot_ground_contact"],
          "geom_ids": [16, 34],
          "command_name": "twist",
          "command_threshold": 0.05,
      },
  )

  # # Unused, only here as an example.
  air_time: RewardTerm = term(
      RewardTerm,
      func=mdp.feet_air_time,
      weight=-50.0,
      params={
          "asset_name": "robot",
          "sensor_names": ["left_foot_ground_contact", "right_foot_ground_contact"],
          "target_time": 0.40,
          "contact_threshold": 0.1,
          "command_name": "twist",
          "command_threshold": 0.05,
      },
  )

  feet_stance_time_target: RewardTerm = term(
      RewardTerm,
      func=mdp.feet_stance_time_target,
      weight=500.0,
      params={
          "asset_name": "robot",
          "sensor_names": ["left_foot_ground_contact", "right_foot_ground_contact"],
          "target_time": 0.40,
          "contact_threshold": 0.1,
          "command_name": "twist",
          "command_threshold": 0.05,
      },
  )


@dataclass
class TerminationCfg:
  time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)
  fell_over: DoneTerm = term(
    DoneTerm, func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
  )
  nan_term: DoneTerm = term(DoneTerm,
      func=mdp.nan_detection)

@dataclass
class CurriculumCfg:
  terrain_levels: CurrTerm | None = term(
    CurrTerm, func=mdp.terrain_levels_vel, params={"command_name": "twist"}
  )

  command_vel: CurrTerm | None = term(
    CurrTerm,
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 500 * 24, "range": (-3.0, 3.0)},
      ],
    },
  )

##
# Environment.
##

SIM_CFG = SimulationCfg(
  nan_guard=NanGuardCfg(
        enabled=True,
        buffer_size=100,
        output_dir="/tmp/mjlab/nan_dumps",
        max_envs_to_dump=5,
  ),
  nconmax=35,
  njmax=300,
  ls_parallel=True,
  mujoco=MujocoCfg(
    timestep=0.005,
    iterations=10,
    ls_iterations=20,
  ),
)


@dataclass
class CarryingEnvCfg(ManagerBasedRlEnvCfg):
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  actions: ActionCfg = field(default_factory=ActionCfg)
  rewards: RewardCfg = field(default_factory=RewardCfg)
  events: EventCfg = field(default_factory=EventCfg)
  terminations: TerminationCfg = field(default_factory=TerminationCfg)
  commands: CommandsCfg = field(default_factory=CommandsCfg)
  curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)
  sim: SimulationCfg = field(default_factory=lambda: SIM_CFG)
  viewer: ViewerConfig = field(default_factory=lambda: VIEWER_CONFIG)
  decimation: int = 10  # 50 Hz control frequency.
  episode_length_s: float = 20.0

  def __post_init__(self):
    # Enable curriculum mode for terrain generator.
    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = True
