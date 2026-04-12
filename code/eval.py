# eval.py

from code.configs.satellite_config import CONFIG
from code.envs.satellite import Satellite
from code.models.custom_model import Shared
from code.envs.wrappers.isaacgym_envs_wrapper import IsaacGymWrapper
from code.rewards.satellite_reward import ExponentialStabilizationReward

import isaacgym
import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import argparse

REWARD_MAP = {
    "exp_stabilization": ExponentialStabilizationReward,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training con reward function selezionabile")
    parser.add_argument(
        "--reward-fn",
        choices=list(REWARD_MAP.keys()),
        help="Which RewardFunction?"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if CONFIG["set_seed"]:
        set_seed(CONFIG["seed"])
    
    #################################################################################

    env = Satellite(
        cfg=CONFIG,
        rl_device=CONFIG["rl_device"],
        sim_device=CONFIG["sim_device"],
        graphics_device_id=CONFIG["graphics_device_id"],
        headless=CONFIG["headless"],
        reward_fn=REWARD_MAP[args.reward_fn]()
    )
    
    env = IsaacGymWrapper(env)

    memory = RandomMemory(memory_size=CONFIG["rl"]["memory"]["rollouts"], num_envs=env.num_envs, device=env.device)

    models = {}
    models["policy"] = Shared(env.state_space, env.action_space, env.device)
    models["value"] = models["policy"]  # Shared model for policy and value
   
    CONFIG["rl"]["PPO"]["state_preprocessor_kwargs"] = {
        "size": env.state_space, "device": env.device
    }
    CONFIG["rl"]["PPO"]["value_preprocessor_kwargs"] = {
        "size": 1, "device": env.device
    }

    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo.update(CONFIG["rl"]["PPO"])
   
    agent = PPO(models=models,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.state_space,
            action_space=env.action_space,
            device=env.device)
    
    agent.load("/home/andreaberti/Satellite-Control-Baseline/runs/26-04-10_21-06-21-725698_PPO/checkpoints/best_agent.pt")
    trainer = SequentialTrainer(cfg=CONFIG["rl"]["trainer"], env=env, agents=agent)

    trainer.eval()

    #################################################################################

if __name__ == "__main__":
    main()