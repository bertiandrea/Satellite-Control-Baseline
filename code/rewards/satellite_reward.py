# satellite_reward.py

from code.utils.satellite_util import quat_diff_rad

import isaacgym #BugFix
import torch

from abc import ABC, abstractmethod
import math

class RewardFunction(ABC):

    @abstractmethod
    def compute(self,
                 quats: torch.Tensor,
                 ang_vels: torch.Tensor,
                 ang_accs: torch.Tensor,
                 goal_quat: torch.Tensor,
                 actions: torch.Tensor
                 ) -> torch.Tensor:
        """
        Compute reward given state and actions.
        Must be implemented by subclasses.
        """
        pass

class ExponentialStabilizationReward(RewardFunction):
    """
    Exponential stabilization reward with bonus when within goal radius.
    """
    def __init__(self,
                 scale=0.14 * 2.0 * math.pi,
                 bonus=9.0,
                 goal_deg=0.005):
        super().__init__()
        self.scale = scale
        self.bonus = bonus
        self.goal_rad = math.radians(goal_deg)
        self.prev_phi = None
        self.lambda_u = 0.00005

    def compute(self, quats, ang_vels, ang_accs, goal_quat, actions):
        phi = quat_diff_rad(quats, goal_quat)
        
        exp_term = torch.exp(-phi / self.scale)
        if self.prev_phi is None:
            r = exp_term
        else:
            delta = phi - self.prev_phi
            r = torch.where(delta > 0.0, exp_term - 1.0, exp_term)

        self.prev_phi = phi.clone()

        u_norm_sq = torch.sum(actions ** 2, dim=-1)     # ||u||^2 per env
        r_effort = self.lambda_u * u_norm_sq

        bonus = (phi <= self.goal_rad).float() * self.bonus

        reward = r - r_effort + bonus

        return reward
