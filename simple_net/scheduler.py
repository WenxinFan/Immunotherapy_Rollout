# The copyright for this material is held by Volkswagen Group.
# This code is exclusively for private project use and is not permitted to be published.
# When referencing this work, please cite the following:
# [1] Klushyn, A., Chen, N., Kurle, R., Cseke, B., & van der Smagt, P. (2019). 
# Learning hierarchical priors in VAEs. NeurIPS.
# [2] Chen, N., van der Smagt, P., & Cseke, B. (2022). 
# Local Distance Preserving Auto-encoders using Continuous kNN Graphs. 
# Topological, Algebraic and Geometric Learning Workshops.

from typing import List, Optional
import numpy as np

CLIP_LAM_MAX = 1e4


class ConstantScheduler:
    def __init__(
        self,
        lam: float = 1.0,
    ) -> None:

        self.lam = lam

    def __call__(self, loss: float) -> float:

        return self.lam


class ConstrainedExponentialSchedulerMaLagrange:
    """
    annealing_rate: the rate for updating lam
    constraint_bound: the desired loss value
    """
    def __init__(
        self,
        constraint_bound: float,
        annealing_rate: float,
        start_lam: float = 1.0,
        alpha: float = 0.5,
        lower_bound_lam: float = 0.0,
        adapt_after_first_satisfied: bool = False,
        clip_lam_max: float = CLIP_LAM_MAX,
    ) -> None:
        self.constraint_bound = constraint_bound
        self.annealing_rate = annealing_rate
        self.start_lam = start_lam
        self.lam = start_lam
        self.alpha = alpha

        self.constraint_ma: Optional[float] = None

        self.adapt_after_first_satisfied = adapt_after_first_satisfied
        self.constraint_first_satisfied = False

        self.lower_bound_lam = lower_bound_lam
        self.gamma = self.lam - self.lower_bound_lam
        self.clip_lam_max = clip_lam_max

    def update_lam(self):
        self.gamma = self.gamma * np.exp(self.annealing_rate * self.constraint_ma)

    def __call__(self, loss: float) -> float:
        constraint = loss - self.constraint_bound
        if self.adapt_after_first_satisfied:
            if not self.constraint_first_satisfied and constraint < 0:
                self.constraint_first_satisfied = True

        # moving average
        if self.constraint_ma is None:
            self.constraint_ma = constraint
        else:
            self.constraint_ma = (
                1 - self.alpha
            ) * self.constraint_ma + self.alpha * constraint

        if self.adapt_after_first_satisfied:
            if self.constraint_first_satisfied:
                self.update_lam()
        else:
            self.update_lam()

        self.gamma = float(
            np.clip(self.gamma, 0.0 - self.lower_bound_lam, self.clip_lam_max)
        )
        self.lam = self.gamma + self.lower_bound_lam

        return self.lam
