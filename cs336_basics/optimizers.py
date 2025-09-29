import torch
from torch.optim import Optimizer
from typing import Optional, Callable
import math


class MyAdamW(Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float, float] = (0.9, 0.999), weight_decay=0.0, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = {'lr':lr, 'betas': betas, 'weight_decay': weight_decay, 'eps': eps}
        super().__init__(params, defaults)
        self.eps = eps

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)

                m = state['m']
                v = state['v']

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                state['step'] += 1
                t = state['step']
                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                step_size = lr * math.sqrt(bias_c2) / bias_c1
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-step_size)

                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

        return loss
