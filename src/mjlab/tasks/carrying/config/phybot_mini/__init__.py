import gymnasium as gym

gym.register(
  id="Mjlab-Carrying-Rough-Phybot-C1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:PhybotC1RoughCarryEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:PhybotC1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Carrying-Rough-Phybot-C1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:PhybotC1RoughCarryEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:PhybotC1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Carrying-Flat-Phybot-C1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PhybotC1FlatCarryEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:PhybotC1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Carrying-Flat-Phybot-C1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PhybotC1FlatCarryEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:PhybotC1PPORunnerCfg",
  },
)
