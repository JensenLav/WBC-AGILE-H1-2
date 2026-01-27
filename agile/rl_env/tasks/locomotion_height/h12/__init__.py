import gymnasium as gym

from . import agents

###########
# RL envs #
###########

gym.register(
    id="Velocity-Height-H12-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_height_env_cfg:H12LowerVelocityHeightEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H12VelocityHeightPpoRunnerCfg",
    },
)


################################
# Teacher-Student Distillation #
################################

gym.register(
    id="Velocity-Height-H12-Distillation-Recurrent-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_height_env_cfg:H12VelocityHeightRecurrentStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H12VelocityHeightDistillationRecurrentRunnerCfg",
    },
)

gym.register(
    id="Velocity-Height-H12-Distillation-History-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_height_env_cfg:H12VelocityHeightHistoryStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H12VelocityHeightDistillationHistoryRunnerCfg",
    },
)
