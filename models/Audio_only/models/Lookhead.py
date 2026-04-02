# lookahead.py
import torch
from torch.optim import Optimizer
import copy

class Lookahead(Optimizer):
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        assert 0.0 < alpha <= 1.0
        assert k >= 1
        self.optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_counter = 0
        self.param_groups = self.optimizer.param_groups

        # 缓存“慢权重”
        self.slow_weights = [[p.clone().detach() for p in group['params']]
                             for group in self.param_groups]
        for w in self.slow_weights:
            for p in w:
                p.requires_grad = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter % self.k == 0:
            for group_idx, group in enumerate(self.param_groups):
                for p_idx, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    slow = self.slow_weights[group_idx][p_idx]
                    # 快权重向慢权重靠近
                    slow.data += self.alpha * (p.data - slow.data)
                    p.data.copy_(slow.data)

        return loss

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'slow_weights': self.slow_weights,
            'step_counter': self.step_counter
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.slow_weights = state_dict['slow_weights']
        self.step_counter = state_dict['step_counter']
