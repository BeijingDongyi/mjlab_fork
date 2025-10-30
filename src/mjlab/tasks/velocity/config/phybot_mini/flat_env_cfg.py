from dataclasses import dataclass

from mjlab.tasks.velocity.config.phybot_mini.rough_env_cfg import (
  PhybotC1RoughEnvCfg,
)


@dataclass
class PhybotC1FlatEnvCfg(PhybotC1RoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    assert self.scene.terrain is not None
    self.scene.terrain.terrain_type = "plane"
    self.scene.terrain.terrain_generator = None
    self.curriculum.terrain_levels = None

    self.curriculum.command_vel = None

    assert self.events.push_robot is not None
    self.events.push_robot.params["velocity_range"] = {
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
    }


@dataclass
class PhybotC1FlatEnvCfg_PLAY(PhybotC1FlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)
