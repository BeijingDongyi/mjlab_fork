import gymnasium as gym

gym.register(
  id="Mjlab-Tracking-Flat-Phybot-C1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PhybotC1FlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:PhybotC1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Phybot-C1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PhybotC1FlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:PhybotC1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Phybot-C1-No-State-Estimation",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PhybotC1FlatNoStateEstimationEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:PhybotC1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Phybot-C1-No-State-Estimation-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PhybotC1FlatNoStateEstimationEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:PhybotC1FlatPPORunnerCfg",
  },
)
