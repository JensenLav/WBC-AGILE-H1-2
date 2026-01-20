import gymnasium as gym

from . import agents

###########
# RL envs #
###########

gym.register(
    id="Velocity-H12-History-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_history_env_cfg:H12LowerVelocityHistoryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H12VelocityPpoRunnerCfg",
    },
)
