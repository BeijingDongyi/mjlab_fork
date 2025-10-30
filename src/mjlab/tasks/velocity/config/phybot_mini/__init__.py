import gymnasium as gym

gym.register(
  id="Mjlab-Velocity-Rough-Phybot-C1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:PhybotC1RoughEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:PhybotC1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Rough-Phybot-C1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:PhybotC1RoughEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:PhybotC1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Phybot-C1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PhybotC1FlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:PhybotC1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Phybot-C1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PhybotC1FlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:PhybotC1PPORunnerCfg",
  },
)
