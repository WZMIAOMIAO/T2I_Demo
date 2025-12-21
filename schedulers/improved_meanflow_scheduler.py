from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Params:
    # Noise distribution
    P_mean: float = -0.4
    P_std: float = 1.0

    # cfg sample params
    w_min: float = 1.0
    w_max: float = 8.0
    beta: float = 2.0

    rf_ratio: float = 0.5


class IMFScheduler:
    def __init__(self, device, params: Optional[Params] = None) -> None:
        if params is None:
            params = Params()
        self.params: Params = params
        self.device: torch.device = device

    def _logit_normal_dist(self, bsz: int):
        x = torch.randn(size=(bsz,))
        x = torch.sigmoid(x * self.params.P_std + self.params.P_mean)
        return x

    def sample_t_r_cfg(self, bsz: int):
        t = self._logit_normal_dist(bsz)
        r = self._logit_normal_dist(bsz)
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        mask = torch.rand(size=(bsz,)) < self.params.rf_ratio
        r = torch.where(mask, t, r)

        t_min = torch.rand(size=(bsz,)) * 0.5        # sample in [0, 0.5]
        t_max = torch.rand(size=(bsz,)) * 0.5 + 0.5  # sample in [0.5, 1.0]
        # sample w ~ p(w) ∝ w^(-beta)
        assert self.params.beta != 1  # TODO: beta == 1
        u = torch.rand(size=(bsz,))
        term_min = self.params.w_min ** (1.0 - self.params.beta)
        term_max = self.params.w_max ** (1.0 - self.params.beta)
        w = (term_min + u * (term_max - term_min)) ** (1.0 / (1.0 - self.params.beta))
        mask = torch.logical_or(t < t_min, t > t_max)
        w[mask] = 1.0
        return t, r, w, t_min, t_max

    def compute_loss(self, model, bsz: int):
        pass


if __name__ == '__main__':
    scheduler = IMFScheduler(torch.device("cpu"))
    scheduler.sample_t_r_cfg(4)
