from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.phybot_mini.phybot_mini_constants_for_tracking import phybot_ACTION_SCALE, phybot_ROBOT_CFG,Carrying_Box_CFG
from mjlab.tasks.box_task.tracking_env_cfg import TrackingEnvCfg
from mjlab.utils.spec_config import ContactSensorCfg


@dataclass
class PhybotC1FlatEnvCfg(TrackingEnvCfg):
  def __post_init__(self):
    self.scene.entities = {"robot": replace(phybot_ROBOT_CFG),
                           "box":replace(Carrying_Box_CFG)}

    self_collision_cfg = ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    )
    self.scene.sensors = (self_collision_cfg,)

    self.actions.joint_pos.scale = phybot_ACTION_SCALE

    self.commands.motion.anchor_body_name = "waist_yaw"
    self.commands.motion.body_names = [
      "base_link",
      "left_hip_roll",
      "left_knee",
      "left_ankle_roll",
      "right_hip_roll",
      "right_knee",
      "right_ankle_roll",
      "waist_yaw",
      "left_shoulder_roll",
      "left_elbow_pitch",
      "right_shoulder_roll",
      "right_elbow_pitch",
    ]

    self.events.foot_friction.params["asset_cfg"].geom_names = [
      r"^(left|right)_foot[1-7]_collision$"
    ]
    self.events.base_com.params["asset_cfg"].body_names = "waist_yaw"

    self.terminations.ee_body_pos.params["body_names"] = [
      "left_ankle_roll",
      "right_ankle_roll",
      "left_elbow_pitch",
      "right_elbow_pitch",
    ]

    self.viewer.body_name = "waist_yaw"


@dataclass
class PhybotC1FlatNoStateEstimationEnvCfg(PhybotC1FlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.motion_anchor_pos_b = None
    self.observations.policy.base_lin_vel = None


@dataclass
class PhybotC1FlatEnvCfg_PLAY(PhybotC1FlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.enable_corruption = False
    self.events.push_robot = None

    # Disable RSI randomization.
    self.commands.motion.pose_range = {}
    self.commands.motion.velocity_range = {}

    # Disable adaptive sampling to play through motion from start to finish.
    self.commands.motion.disable_adaptive_sampling = True

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)


@dataclass
class PhybotC1FlatNoStateEstimationEnvCfg_PLAY(PhybotC1FlatNoStateEstimationEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.enable_corruption = False
    self.events.push_robot = None

    # Disable RSI randomization.
    self.commands.motion.pose_range = {}
    self.commands.motion.velocity_range = {}

    # Disable adaptive sampling to play through motion from start to finish.
    self.commands.motion.disable_adaptive_sampling = True

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)
