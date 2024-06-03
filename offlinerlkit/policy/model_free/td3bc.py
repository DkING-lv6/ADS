import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, Union, Tuple, Callable
from offlinerlkit.policy import TD3Policy
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.scaler import StandardScaler
import learn2learn as l2l
from copy import deepcopy


class TD3BCPolicy(TD3Policy):
    """
    TD3+BC <Ref: https://arxiv.org/abs/2106.06860>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        max_action: float = 1.0,
        exploration_noise: Callable = GaussianNoise,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        update_actor_freq: int = 2,
        alpha: float = 2.5,
        scaler: StandardScaler = None
    ) -> None:

        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            max_action=max_action,
            exploration_noise=exploration_noise,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            update_actor_freq=update_actor_freq
        )

        self._alpha = alpha
        self.scaler = scaler

        self.meta1_model = l2l.algorithms.MAML(self.critic1, lr=0.00005)  # 元模型
        self.meta2_model = l2l.algorithms.MAML(self.critic2, lr=0.00005)  # 元模型
    
    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.scaler is not None:
            obs = self.scaler.transform(obs)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()
        if not deterministic:
            action = action + self.exploration_noise(action.shape)
            action = np.clip(action, -self._max_action, self._max_action)
        return action
    
    def learn(self, batch: Dict, batch_t : Dict, batch_all : Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        obss_t, actions_t, next_obss_t, rewards_t, terminals_t = batch_t["observations"], batch_t["actions"], \
            batch_t["next_observations"], batch_t["rewards"], batch_t["terminals"]
        obss_all, actions_all, next_obss_all, rewards_all, terminals_all = batch_all["observations"], batch_all[
            "actions"], batch_all["next_observations"], batch_all["rewards"], batch_all["terminals"]

        # ---------------------- st ------------------------------
        new_critic1 = self.meta1_model.clone()
        new_critic2 = self.meta2_model.clone()
        q1, q2 = new_critic1(obss_t, actions_t), new_critic2(obss_t, actions_t)
        with torch.no_grad():
            noise = (torch.randn_like(actions_t) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            next_actions = (self.actor_old(next_obss_t) + noise).clamp(-self._max_action, self._max_action)
            next_q = torch.min(self.critic1_old(next_obss_t, next_actions), self.critic2_old(next_obss_t, next_actions))
            target_q = rewards_t + self._gamma * (1 - terminals_t) * next_q
        critic1_loss_t = ((q1 - target_q).pow(2)).mean()
        critic2_loss_t = ((q2 - target_q).pow(2)).mean()
        new_critic1.adapt(critic1_loss_t)
        new_critic2.adapt(critic2_loss_t)
        # ----------------------end -------------------------------
        # update critic
        q1, q2 = new_critic1(obss, actions), new_critic2(obss, actions)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            next_actions = (self.actor_old(next_obss) + noise).clamp(-self._max_action, self._max_action)
            next_q = torch.min(self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions))
            target_q = rewards + self._gamma * (1 - terminals) * next_q
        
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        loss1 = critic1_loss * 0.0001 + critic1_loss_t
        loss2 = critic2_loss * 0.0001 + critic2_loss_t
        self.critic1_optim.zero_grad()
        loss1.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        loss2.backward()
        self.critic2_optim.step()

        # update actor
        if self._cnt % self._freq == 0:
            a = self.actor(obss)
            q = self.critic1(obss, a)
            lmbda = self._alpha / q.abs().mean().detach()
            actor_loss = -lmbda * q.mean() + ((a - actions).pow(2)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self._last_actor_loss = actor_loss.item()
            self._sync_weight()
        
        self._cnt += 1

        return {
            "loss/actor": self._last_actor_loss,
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

    def learnCopy(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
                                                       batch["next_observations"], batch["rewards"], batch["terminals"]

        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            next_actions = (self.actor_old(next_obss) + noise).clamp(-self._max_action, self._max_action)
            next_q = torch.min(self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions))
            target_q = rewards + self._gamma * (1 - terminals) * next_q

        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        if self._cnt % self._freq == 0:
            a = self.actor(obss)
            q = self.critic1(obss, a)
            lmbda = self._alpha / q.abs().mean().detach()
            actor_loss = -lmbda * q.mean() + ((a - actions).pow(2)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self._last_actor_loss = actor_loss.item()
            self._sync_weight()

        self._cnt += 1

        return {
            "loss/actor": self._last_actor_loss,
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

    def learn_data_split1(self, batch: Dict):
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
                                                       batch["next_observations"], batch["rewards"], batch["terminals"]
        q1= self.critic1(obss, actions)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            next_actions = (self.actor_old(next_obss) + noise).clamp(-self._max_action, self._max_action)
            next_q = torch.min(self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions))
            target_q = rewards + self._gamma * (1 - terminals) * next_q

        critic1_loss = ((q1 - target_q).pow(2)).mean()

        self.critic1_optim.zero_grad()
        grad = torch.autograd.grad(critic1_loss, self.critic1.parameters())

        return {
            "loss": critic1_loss.item(),
            "grad": grad
        }