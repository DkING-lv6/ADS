import numpy as np
import torch
import torch.nn as nn
import gym
import os

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from offlinerlkit.policy import SACPolicy
from copy import deepcopy
import learn2learn as l2l
from offlinerlkit.nets import MLP
from offlinerlkit.modules import Critic

class CQLDATASPLIT1Policy(SACPolicy):
    """
    Conservative Q-Learning <Ref: https://arxiv.org/abs/2006.04779>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Space,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        cql_weight: float = 1.0,
        temperature: float = 1.0,
        max_q_backup: bool = False,
        deterministic_backup: bool = True,
        with_lagrange: bool = True,
        lagrange_threshold: float = 10.0,
        cql_alpha_lr: float = 1e-4,
        num_repeart_actions:int = 10,
        critic_lr: float = 3e-4,
        max_weight: int = 10,
        min_weight: int = -10,
        file_name: str = "dc"
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
            alpha=alpha
        )

        self.action_space = action_space
        self._cql_weight = cql_weight
        self._temperature = temperature
        self._max_q_backup = max_q_backup
        self._deterministic_backup = deterministic_backup
        self._with_lagrange = with_lagrange
        self._lagrange_threshold = lagrange_threshold

        self.cql_log_alpha = torch.zeros(1, requires_grad=True, device=self.actor.device)
        self.cql_alpha_optim = torch.optim.Adam([self.cql_log_alpha], lr=cql_alpha_lr)

        self._num_repeat_actions = num_repeart_actions
        self.critic_lr = critic_lr
        self.max_w = max_weight
        self.min_w = min_weight
        self.file_name = file_name

        self.meta1_model = l2l.algorithms.MAML(self.critic1, lr=0.00005)  # 元模型
        self.meta1_opt = torch.optim.Adam(self.meta1_model.parameters(), lr=0.0005)

        self.meta2_model = l2l.algorithms.MAML(self.critic2, lr=0.00005)  # 元模型
        self.meta2_opt = torch.optim.Adam(self.meta2_model.parameters(), lr=0.0005)

    def calc_pi_values(
        self,
        obs_pi: torch.Tensor,
        obs_to_pred: torch.Tensor,
        critic1,
        critic2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        act, log_prob = self.actforward(obs_pi)

        q1 = critic1(obs_to_pred, act)
        
        q2 = critic2(obs_to_pred, act)

        return q1 - log_prob.detach(), q2 - log_prob.detach()

    def calc_random_values(
        self,
        obs: torch.Tensor,
        random_act: torch.Tensor,
        critic1,
        critic2
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        q1 = critic1(obs, random_act)
        q2 = critic2(obs, random_act)

        log_prob1 = np.log(0.5**random_act.shape[-1])
        log_prob2 = np.log(0.5**random_act.shape[-1])

        return q1 - log_prob1, q2 - log_prob2

    def get_q(self, obs, act):
        return self.critic1(obs, act)

    def learn(self, batch: Dict, batch_t : Dict, batch_all : Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        batch_size = obss.shape[0]
        obss_t, actions_t, next_obss_t, rewards_t, terminals_t = batch_t["observations"], batch_t["actions"], \
             batch_t["next_observations"], batch_t["rewards"], batch_t["terminals"]
        batch_size_t = obss_t.shape[0]
        obss_all, actions_all, next_obss_all, rewards_all, terminals_all = batch_all["observations"], batch_all["actions"], \
            batch_all["next_observations"], batch_all["rewards"], batch_all["terminals"]

        # print([w for w in self.actor.parameters()])
        # update actor
        a, log_probs = self.actforward(obss_all)
        q1a,q2a = self.critic1(obss_all, a) ,self.critic2(obss_all, a)
        actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
        # ----------------------------- train_batch -- space  -------------------------------------------
        new_critic1 = self.meta1_model.clone()
        new_critic2 = self.meta2_model.clone()

        with torch.no_grad():
            next_actions_t, next_log_probs_t = self.actforward(next_obss_t)
            next_q_t = torch.min(
                self.critic1_old(next_obss_t, next_actions_t.detach()),
                self.critic2_old(next_obss_t, next_actions_t.detach())
            )
            if not self._deterministic_backup:
                next_q_t -= self._alpha * next_log_probs_t

        target_q_t = rewards_t + self._gamma * (1 - terminals_t) * next_q_t
        q1_t, q2_t = new_critic1(obss_t, actions_t), new_critic2(obss_t, actions_t)
        q1_t_copy = self.critic1(obss_t, actions_t)
        critic1_loss_t = ((q1_t - target_q_t).pow(2)).mean()
        critic2_loss_t = ((q2_t - target_q_t).pow(2)).mean()
        critic1_loss_t_copy = ((q1_t_copy - target_q_t).pow(2)).mean()

        # compute conservative loss
        random_actions_t = torch.FloatTensor(
            batch_size_t * self._num_repeat_actions, actions.shape[-1]
        ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
        # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
        tmp_obss_t = obss_t.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size_t * self._num_repeat_actions, obss_t.shape[-1])
        tmp_next_obss_t = next_obss_t.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size_t * self._num_repeat_actions, obss_t.shape[-1])

        # -----------------------
        act, log_prob = self.actforward(tmp_obss_t)
        q1ct = new_critic1(tmp_obss_t, act)
        q2c = new_critic2(tmp_obss_t, act)
        q1c_copy = self.critic1(tmp_obss_t, act)
        # -----------------------
        obs_pi_value1, obs_pi_value1_copy = q1ct - log_prob.detach(), q1c_copy - log_prob.detach()
        obs_pi_value2 = q2c - log_prob.detach()
        # -----------------------
        act, log_prob = self.actforward(tmp_next_obss_t)
        q1ct2 = new_critic1(tmp_obss_t, act)
        q2c = new_critic2(tmp_obss_t, act)
        q1c_copy = self.critic1(tmp_obss_t, act)
        # -----------------------
        next_obs_pi_value1, next_obs_pi_value1_copy = q1ct2 - log_prob.detach(), q1c_copy - log_prob.detach()
        next_obs_pi_value2 = q2c - log_prob.detach()
        # ------------------------
        q1ct3 = new_critic1(tmp_obss_t, random_actions_t)
        q2c = new_critic2(tmp_obss_t, random_actions_t)
        q1c_copy = self.critic1(tmp_obss_t, random_actions_t)

        log_prob1 = np.log(0.5 ** random_actions_t.shape[-1])
        log_prob2 = np.log(0.5 ** random_actions_t.shape[-1])
        # ------------------------
        random_value1, random_value1_copy = q1ct3 - log_prob1, q1c_copy - log_prob1
        random_value2 = q2c - log_prob2

        for value in [
            obs_pi_value1, obs_pi_value1_copy, next_obs_pi_value1, next_obs_pi_value1_copy,
            random_value1, random_value1_copy, obs_pi_value2, next_obs_pi_value2, random_value2
        ]:
            value.reshape(batch_size_t, self._num_repeat_actions, 1)

        # cat_q shape: (batch_size, 3 * num_repeat, 1)
        cat_q1_t = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)
        cat_q1_t_copy = torch.cat([obs_pi_value1_copy, next_obs_pi_value1_copy, random_value1_copy], 1)
        cat_q2 = torch.cat([obs_pi_value2, next_obs_pi_value2, random_value2], 1)
        cat_qt = torch.cat([q1ct,q1ct2,q1ct3],1)

        conservative_loss1_t = \
            torch.logsumexp(cat_q1_t / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q1_t.mean() * self._cql_weight
        conservative_loss1_t_copy = \
            torch.logsumexp(cat_q1_t_copy / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q1_t_copy.mean() * self._cql_weight
        conservative_loss2_t = \
            torch.logsumexp(cat_q2 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q2_t.mean() * self._cql_weight

        critic11_loss_t = critic1_loss_t + conservative_loss1_t
        critic22_loss_t = critic2_loss_t + conservative_loss2_t

        c_q1_dif = (cat_qt.mean() - q1_t.mean().item())
        if c_q1_dif < 0:
            critic11_loss_t = critic1_loss_t
            critic22_loss_t = critic2_loss_t

        critic1_loss_t_copy = critic1_loss_t_copy + conservative_loss1_t_copy

        new_critic1.adapt(critic11_loss_t)
        new_critic2.adapt(critic22_loss_t)

        #--------------------------------- end -----------------------------------------
        # update actor
        copy_actor = deepcopy(self.actor)
        # print("\n--------\n")
        # print([m.detach() for m in self.actor.parameters()])
        copy_critic1 = deepcopy(self.critic1)
        copy_critic2 = deepcopy(self.critic2)
        w1=[m.detach() for m in copy_actor.parameters()]
        copy_w1 = [m for m in new_critic1.module.parameters()]
        copy_w2 = [m for m in new_critic2.module.parameters()]
        self.copy_critic(copy_critic1,copy_w1)
        self.copy_critic(copy_critic2,copy_w2)
        actor_optim = torch.optim.Adam(copy_actor.parameters(), lr=1e-4)
        a, log_probs = self.copy_act(copy_actor,obss_all)
        q1a, q2a = copy_critic1(obss_all, a), copy_critic2(obss_all, a)
        actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()
        # print([m.detach() for m in self.actor.parameters()])
        # print("\n--------\n")
        #
        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            log_alpha = deepcopy(self._log_alpha)
            alpha_optim = torch.optim.Adam([log_alpha], lr=1e-4)

            alpha_loss = -(log_alpha * log_probs).mean()
            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()
            self._alpha = log_alpha.detach().exp()
        #---------------------------------------------------------------------------------
        # compute td error
        with torch.no_grad():
            # next_actions, next_log_probs = self.actforward(next_obss)
            next_actions, next_log_probs = self.copy_act(copy_actor, next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions.detach()),
                self.critic2_old(next_obss, next_actions.detach())
            )
            if not self._deterministic_backup:
                next_q -= self._alpha * next_log_probs

        target_q = rewards + self._gamma * (1 - terminals) * next_q
        q1, q2 = new_critic1(obss, actions), new_critic2(obss, actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        # compute conservative loss
        random_actions = torch.FloatTensor(
            batch_size * self._num_repeat_actions, actions.shape[-1]
        ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
        # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
        tmp_obss = obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        tmp_next_obss = next_obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])

        # -----------------------
        act, log_prob = self.copy_act(copy_actor, tmp_obss)
        q1c = new_critic1(tmp_obss, act)
        q2c = new_critic2(tmp_obss, act)
        # -----------------------
        obs_pi_value1, obs_pi_value2 = q1c - log_prob.detach(), q2c - log_prob.detach()
        # -----------------------
        act, log_prob = self.copy_act(copy_actor, tmp_next_obss)
        q1c = new_critic1(tmp_obss, act)
        q2c = new_critic2(tmp_obss, act)
        # -----------------------
        next_obs_pi_value1, next_obs_pi_value2 = q1c - log_prob.detach(), q2c - log_prob.detach()
        # ------------------------
        q1c = new_critic1(tmp_obss, random_actions)
        q2c = new_critic2(tmp_obss, random_actions)

        log_prob1 = np.log(0.5 ** random_actions.shape[-1])
        log_prob2 = np.log(0.5 ** random_actions.shape[-1])
        # ------------------------
        random_value1, random_value2 = q1c - log_prob1, q2c - log_prob2

        for value in [
            obs_pi_value1, obs_pi_value2, next_obs_pi_value1, next_obs_pi_value2,
            random_value1, random_value2
        ]:
            value.reshape(batch_size, self._num_repeat_actions, 1)

        # cat_q shape: (batch_size, 3 * num_repeat, 1)
        cat_q1 = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)
        cat_q2 = torch.cat([obs_pi_value2, next_obs_pi_value2, random_value2], 1)

        conservative_loss1 = \
            torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q1.mean() * self._cql_weight
        conservative_loss2 = \
            torch.logsumexp(cat_q2 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q2.mean() * self._cql_weight
        # print(str((cat_q1.mean() - q1.mean()).item()))

        critic_loss1 = critic1_loss + conservative_loss1
        critic_loss2 = critic2_loss + conservative_loss2

        loss2 = critic1_loss * 0.00001 + critic11_loss_t
        loss3 = critic2_loss * 0.00001 + critic22_loss_t

        self.critic1_optim.zero_grad()
        loss2.backward(retain_graph=True)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        loss3.backward()
        self.critic2_optim.step()

        # --------------------------------
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        # --------------------------------
        self.actor = deepcopy(copy_actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": loss2.item(),
            "loss/critic2": loss3.item()
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = 0
            result["alpha"] = self._alpha.item()
        return result

    def learn_policy(self,batch: Dict):
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
                                                       batch["next_observations"], batch["rewards"], batch["terminals"]

        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)
        actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            # o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
            o.data.copy_(n.data)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            # o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
            o.data.copy_(n.data)

        result = {
            "loss/actor": actor_loss.item(),
        }
        return result

    # 计算损失和梯度，但是不执行更新操作
    def learn_data_split1(self, batch: Dict) :
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        batch_size = obss.shape[0]

        # compute td error
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions),
                self.critic2_old(next_obss, next_actions)
            )
            if not self._deterministic_backup:
                next_q -= self._alpha * next_log_probs

        target_q = rewards + self._gamma * (1 - terminals) * next_q
        q1 = self.critic1(obss, actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        # compute conservative loss
        random_actions = torch.FloatTensor(
            batch_size * self._num_repeat_actions, actions.shape[-1]
        ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
        # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
        tmp_obss = obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        tmp_next_obss = next_obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        
        #-----------------------
        act, log_prob = self.actforward(tmp_obss)
        q1c = self.critic1(tmp_obss, act)
        #-----------------------
        obs_pi_value1 = q1c - log_prob.detach()
        # -----------------------
        act, log_prob = self.actforward(tmp_next_obss)
        q1c = self.critic1(tmp_obss, act)
        # -----------------------
        next_obs_pi_value1 = q1c - log_prob.detach()
        #------------------------
        q1c = self.critic1(tmp_obss, random_actions)

        log_prob1 = np.log(0.5**random_actions.shape[-1])
        #------------------------
        random_value1 = q1c - log_prob1


        for value in [
            obs_pi_value1, next_obs_pi_value1,
            random_value1
        ]:
            value.reshape(batch_size, self._num_repeat_actions, 1)
        
        # cat_q shape: (batch_size, 3 * num_repeat, 1)
        cat_q1 = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)

        conservative_loss1 = \
            torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q1.mean() * self._cql_weight
        
        critic2_loss = critic1_loss + conservative_loss1


        self.critic1_optim.zero_grad()
        grad = torch.autograd.grad(critic2_loss, self.critic1.parameters())

        return {
            "loss": critic2_loss.item(),
            "grad": grad
        }

    def learn_data_split2(self, batch: Dict):
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
                                                       batch["next_observations"], batch["rewards"], batch["terminals"]
        batch_size = obss.shape[0]

        # compute td error
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions),
                self.critic2_old(next_obss, next_actions)
            )
            if not self._deterministic_backup:
                next_q -= self._alpha * next_log_probs

        target_q = rewards + self._gamma * (1 - terminals) * next_q
        q2 = self.critic2(obss, actions)
        critic2_loss = ((q2 - target_q).pow(2)).mean()
        # compute conservative loss
        random_actions = torch.FloatTensor(
            batch_size * self._num_repeat_actions, actions.shape[-1]
        ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
        # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
        tmp_obss = obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        tmp_next_obss = next_obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])

        # -----------------------
        act, log_prob = self.actforward(tmp_obss)
        q2c = self.critic2(tmp_obss, act)
        # -----------------------
        obs_pi_value2 = q2c - log_prob.detach()
        # -----------------------
        act, log_prob = self.actforward(tmp_next_obss)
        q2c = self.critic2(tmp_obss, act)
        # -----------------------
        next_obs_pi_value2 = q2c - log_prob.detach()
        # ------------------------
        q2c = self.critic2(tmp_obss, random_actions)

        log_prob2 = np.log(0.5 ** random_actions.shape[-1])
        # ------------------------
        random_value2 = q2c - log_prob2

        for value in [
            obs_pi_value2, next_obs_pi_value2, random_value2
        ]:
            value.reshape(batch_size, self._num_repeat_actions, 1)

        # cat_q shape: (batch_size, 3 * num_repeat, 1)
        cat_q2 = torch.cat([obs_pi_value2, next_obs_pi_value2, random_value2], 1)

        conservative_loss2 = \
            torch.logsumexp(cat_q2 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q2.mean() * self._cql_weight

        critic2_loss = critic2_loss + conservative_loss2

        self.critic2_optim.zero_grad()
        grad = torch.autograd.grad(critic2_loss, self.critic2.parameters())

        return {
            "loss": critic2_loss.item(),
            "grad": grad
        }


    def get_variance(self, obs, acts, update_type = 1):
        std_Q_list1, std_Q_list2 = [], []
        L_target_Q1, L_target_Q2 = [], []
        for index in range(self.num_ensemble):
            if update_type == 0:
                target_Q1 = self.critic1(obs, acts)
                target_Q2 = self.critic2(obs, acts)
            else:
                target_Q1 = self.critic1_old(obs, acts)
                target_Q2 = self.critic2_old(obs, acts)
            L_target_Q1.append(target_Q1)
            L_target_Q2.append(target_Q2)

        std_Q_list1.append(torch.stack(L_target_Q1).std(axis=0).detach())
        return torch.stack(std_Q_list1).squeeze(0)

    def learnCopy(self, batch: Dict, cmax=None, cmin=None) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
                                batch["next_observations"], batch["rewards"], batch["terminals"]
        batch_size = obss.shape[0]

        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)
        actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()
        # actor_loss = (self._alpha * log_probs - q1a).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        # compute td error
        if self._max_q_backup:
            with torch.no_grad():
                tmp_next_obss = next_obss.unsqueeze(1) \
                    .repeat(1, self._num_repeat_actions, 1) \
                    .view(batch_size * self._num_repeat_actions, next_obss.shape[-1])
                tmp_next_actions, _ = self.actforward(tmp_next_obss)
                tmp_next_q1 = self.critic1_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                tmp_next_q2 = self.critic2_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                next_q = torch.min(tmp_next_q1, tmp_next_q2)
        else:
            with torch.no_grad():
                next_actions, next_log_probs = self.actforward(next_obss)
                next_q = torch.min(
                    self.critic1_old(next_obss, next_actions),
                    self.critic2_old(next_obss, next_actions)
                )
                # next_q = self.critic1_old(next_obss, next_actions)
                if not self._deterministic_backup:
                    next_q -= self._alpha * next_log_probs

        target_q = rewards + self._gamma * (1 - terminals) * next_q
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        # compute conservative loss
        random_actions = torch.FloatTensor(
            batch_size * self._num_repeat_actions, actions.shape[-1]
        ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
        # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
        tmp_obss = obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        tmp_next_obss = next_obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])

        #-----------------------
        act, log_prob = self.actforward(tmp_obss)
        q1c = self.critic1(tmp_obss, act)
        q2c = self.critic2(tmp_obss, act)
        #-----------------------
        obs_pi_value1, obs_pi_value2 = q1c - log_prob.detach(), q2c - log_prob.detach()
        # -----------------------
        act, log_prob = self.actforward(tmp_next_obss)
        q1c = self.critic1(tmp_obss, act)
        q2c = self.critic2(tmp_obss, act)
        # -----------------------
        next_obs_pi_value1, next_obs_pi_value2 = q1c - log_prob.detach(), q2c - log_prob.detach()
        #------------------------
        q1c = self.critic1(tmp_obss, random_actions)
        q2c = self.critic2(tmp_obss, random_actions)

        log_prob1 = np.log(0.5**random_actions.shape[-1])
        log_prob2 = np.log(0.5**random_actions.shape[-1])
        #------------------------
        random_value1, random_value2 = q1c - log_prob1, q2c - log_prob2

        for value in [
            obs_pi_value1, obs_pi_value2, next_obs_pi_value1, next_obs_pi_value2,
            random_value1, random_value2
        ]:
            value.reshape(batch_size, self._num_repeat_actions, 1)

        # cat_q shape: (batch_size, 3 * num_repeat, 1)
        cat_q1 = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)
        cat_q2 = torch.cat([obs_pi_value2, next_obs_pi_value2, random_value2], 1)

        conservative_loss1 = \
            torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q1.mean() * self._cql_weight
        conservative_loss2 = \
            torch.logsumexp(cat_q2 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q2.mean() * self._cql_weight


        # print("critic1_loss" + str(critic1_loss))
        # print("conservative_loss1" + str(conservative_loss1))
        # print("critic2_loss" + str(critic2_loss))
        # print("conservative_loss2" + str(conservative_loss2))
        critic1_loss = critic1_loss + conservative_loss1
        critic2_loss = critic2_loss + conservative_loss2

        # with open("/media/YQX/OfflineRL-Kit/test4.txt", "a") as f:
        #     f.write("critic1_loss" + str(critic1_loss))
        # if i == 1:
        #     with open("/media/YQX/OfflineRL-Kit/test4.txt", "a") as f:
        #         f.write("\n---------------\n")
        #     break
        # update critic

        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        #--------------------------------
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        #--------------------------------
        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result

    def copy_act(self,actor, obs):
        dist = actor(obs)
        squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        return squashed_action, log_prob

    def copy_critic(self,critic, params):
        i = 0
        for m in critic.backbone.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = params[i]
                m.bias.data = params[i + 1]
                i = i + 2
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data = params[i]
                m.bias.data = params[i + 1]
                i = i + 2
        for m in critic.last.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = params[i]
                m.bias.data = params[i + 1]
                i = i + 2
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data = params[i]
                m.bias.data = params[i + 1]
                i = i + 2