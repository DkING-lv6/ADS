import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional


class Critic(nn.Module):
    def __init__(self, backbone: nn.Module, device: str = "cpu") -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        self.last = nn.Linear(latent_dim, 1).to(device)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        weightM = None
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
            obs = torch.cat([obs, actions], dim=1)
        if weightM is None:
            logits = self.backbone(obs)
        else:
            i = 0
            for m in self.backbone.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data = weightM[i]
                    m.bias.data = weightM[i+1]
                    i = i +2
                if isinstance(m, nn.BatchNorm1d):
                    m.weight.data = weightM[i]
                    m.bias.data = weightM[i + 1]
                    i = i + 2
            logits = self.backbone(obs)
            for m in self.last.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data = weightM[i]
                    m.bias.data = weightM[i+1]
                    i = i +2
                if isinstance(m, nn.BatchNorm1d):
                    m.weight.data = weightM[i]
                    m.bias.data = weightM[i + 1]
                    i = i + 2
        values = self.last(logits)
        return values