# satellite_config.py

from pathlib import Path
import numpy as np

import isaacgym
import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

NUM_ENVS = 4096
N_EPISODES = 16
EPISODE_LENGTH_S = 60.0
HEADLESS = True
PROFILE = False
DEBUG_ARROWS = False
DEBUG_PRINTS = False

CONFIG = {
    # --- seed & devices ----------------------------------------------------
    "set_seed": True,
    "seed": 42,

    "physics_engine": "physx",

    "rl_device": "cuda:0",
    "sim_device": "cuda:0",
    "graphics_device_id": 0,
    "headless": HEADLESS,

    "profile": PROFILE,

    # --- env section -------------------------------------------------------
    "env": {
        "numEnvs": NUM_ENVS,
        "numObservations": 15, # satellite_quats (4) + quat_diff (4) + quat_diff_rad (1) + satellite_angacc (3) + actions (3)
        "numStates": 18, # satellite_quats (4) + quat_diff (4) + quat_diff_rad (1) + satellite_angacc (3) + actions (3) + satellite_angvels (3)
        "numActions": 3,
        "clipActions": 1.0,
        "clipObservations": 10.0,

        "env_spacing": 3.0,

        "episode_length_s": EPISODE_LENGTH_S, # seconds
        "torque_scale": 100.0,
        "debug_arrows": DEBUG_ARROWS,
        "debug_prints": DEBUG_PRINTS,
        
        "asset": {
            "asset_root": str(Path(__file__).resolve().parent.parent),
            "asset_file_name": "satellite.urdf",
            "asset_name": "satellite",
        },
    },

    # --- sim section -------------------------------------------------------
    "sim": {
        "dt": 1.0 / 60.0,
        "gravity": [0.0, 0.0, 0.0],
        "up_axis": "z",
        "use_gpu_pipeline": True,
        "substeps": 2,

        "physx": {
            "use_gpu": True,
        },
    },

    # --- RL / PPO hyper-params --------------------------------------------
    "rl": {
        "PPO": {
            "num_envs": NUM_ENVS,
            "rollouts": 16,
            "learning_epochs": 8,
            "mini_batches": 4,
            
            "learning_rate_scheduler" : KLAdaptiveRL,
            "learning_rate_scheduler_kwargs" : {"kl_threshold": 0.016},
            "state_preprocessor" : RunningStandardScaler,
            "value_preprocessor" : RunningStandardScaler,
            "rewards_shaper" : lambda rewards, timestep, timesteps: rewards * 0.01,

            "discount_factor" : 0.99, #(γ) Future reward discount; balances immediate versus long-term return.
            "learning_rate" : 1e-3, #Step size for optimizer (e.g. Adam) when updating policy and value networks.
            "grad_norm_clip" : 1.0, #Maximum norm value to clip gradients, preventing exploding gradients.
            "ratio_clip" : 0.2, #(ϵ) PPO’s clipping threshold on the policy probability ratio to constrain updates.
            "value_clip" : 0.2, #Clipping range for value function targets to stabilize value updates.
            "clip_predicted_values" : True, #If enabled, clips the new value predictions to lie within the range defined by value_clip around the old predictions.
            "entropy_loss_scale" : 0.00, #Coefficient multiplying the entropy bonus; encourages exploration when > 0.
            "value_loss_scale" : 1.0, #Coefficient weighting the value function loss in the total loss.
            "kl_threshold" : 0, #Optional early-stop threshold on KL divergence between old and new policies (0 disables).
            "lambda" : 0.95, #(λ) GAE parameter for bias–variance trade-off in advantage estimation.

            "random_timesteps" : 0, #Number of initial timesteps with random actions before learning or policy-driven sampling.
            "learning_starts" : 0, #Number of environment steps to collect before beginning any gradient updates.
            
            "experiment": {
                "write_interval": "auto",
                "checkpoint_interval": "auto",
                "directory": "./runs",
                "wandb": False,
            },
        },
        "trainer": {
            "timesteps": int(EPISODE_LENGTH_S / (1.0 / 60.0) * N_EPISODES),
            "disable_progressbar": False,
            "headless": HEADLESS,
        },
        "memory": {
            "rollouts": 16,
        },
    },
}