from __future__ import annotations

import math
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn

import config as cfg
from pinns import LWRPINN


def flux(u: torch.Tensor) -> torch.Tensor:
    return u * (1.0 - u)


class LWRVPINN(nn.Module):
    """
    Weak-form VPINN built on top of LWRPINN for

        u_t + (f(u))_x = 0,   f(u)=u(1-u).

    We use space-time sine test functions on normalized coordinates:
        phi_{mn}(x,t) = sin(m*pi*xi) sin(n*pi*tau)

    and the weak form (integrated by parts in space):
        ∫∫ [u_t * phi - f(u) * phi_x] dx dt = 0
    because phi vanishes at x = x_min, x_max.
    """

    def __init__(
        self,
        layers: List[int],
        activation: Union[str, nn.Module] = "tanh",
        hard_init: bool = True,
        n_fourier: int = 6,
        scale: float = 2.0,
        n_test_x: int = 3,
        n_test_t: int = 3,
    ) -> None:
        super().__init__()
        self.model = LWRPINN(
            layers=layers,
            activation=activation,
            hard_init=hard_init,
            n_fourier=n_fourier,
            scale=scale,
        )
        self.n_test_x = int(n_test_x)
        self.n_test_t = int(n_test_t)

    def forward(self, x: torch.Tensor, t: torch.Tensor, u0_fn=None) -> torch.Tensor:
        return self.model(x, t, u0_fn=u0_fn)

    def weak_residual_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        u0_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        if self.n_test_x < 1 or self.n_test_t < 1:
            raise ValueError("n_test_x and n_test_t must be >= 1")

        x = x.requires_grad_(True)
        t = t.requires_grad_(True)

        u = self.forward(x, t, u0_fn=u0_fn)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]

        xi = (x - cfg.X_MIN) / (cfg.X_MAX - cfg.X_MIN)
        tau = (t - cfg.T_MIN) / (cfg.T_MAX - cfg.T_MIN)

        loss = torch.zeros((), device=x.device)

        for m in range(1, self.n_test_x + 1):
            for n in range(1, self.n_test_t + 1):
                phi = torch.sin(m * math.pi * xi) * torch.sin(n * math.pi * tau)

                phi_x = (
                    (m * math.pi / (cfg.X_MAX - cfg.X_MIN))
                    * torch.cos(m * math.pi * xi)
                    * torch.sin(n * math.pi * tau)
                )

                # Weak form:
                #   ∫∫ [u_t * phi - f(u) * phi_x] dx dt = 0
                integrand = u_t * phi - flux(u) * phi_x

                # Monte Carlo approximation of the space-time integral
                residual_mn = integrand.mean()
                loss = loss + residual_mn.pow(2)

        return loss / float(self.n_test_x * self.n_test_t)