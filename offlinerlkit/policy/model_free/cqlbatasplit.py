import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from offlinerlkit.policy import SACCopyPolicy
from copy import deepcopy
import learn2learn as l2l

class CQLDATASPLITPolicy(SACCopyPolicy):
    """
    Conservative Q-Learning <Ref: https://arxiv.org/abs/2006.04779>
    """

    def __init__(
        self,
        actor: [nn.Module],
        critic1: [nn.Module],
        critic2: [nn.Module],
        actor_optim: [torch.optim.Optimizer],
        critic1_optim: [torch.optim.Optimizer],
        critic2_optim: [torch.optim.Optimizer],
        num_ensemble: int,
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
            num_ensemble,
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

        self.cql_log_alpha = torch.zeros(1, requires_grad=True, device=self.actor[0].device)
        self.cql_alpha_optim = torch.optim.Adam([self.cql_log_alpha], lr=cql_alpha_lr)

        self._num_repeat_actions = num_repeart_actions
        self.critic_lr = critic_lr
        self.max_w = max_weight
        self.min_w = min_weight
        self.file_name = file_name

    def calc_pi_values(
        self,
        obs_pi: torch.Tensor,
        obs_to_pred: torch.Tensor,
        critic1,
        critic2,
        index
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        act, log_prob = self.actforward(obs_pi, index)

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

    def learn(self, batch: Dict, batch_t : Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        batch_size = obss.shape[0]

        obss_t, actions_t, next_obss_t, rewards_t, terminals_t = batch_t["observations"], batch_t["actions"], \
             batch_t["next_observations"], batch_t["rewards"], batch_t["terminals"]
        batch_size_t = obss_t.shape[0]

        for index in range(self.num_ensemble):

            # update actor
            a, log_probs = self.actforward(obss,index)
            q1a, q2a = self.critic1[index](obss, a), self.critic2[index](obss, a)
            actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()

            self.actor_optim[index].zero_grad()
            actor_loss.backward()
            self.actor_optim[index].step()

            if self._is_auto_alpha:
                log_probs = log_probs.detach() + self._target_entropy
                alpha_loss = -(self._log_alpha * log_probs).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self._alpha = self._log_alpha.detach().exp()

            # ----------------------------- train_batch -- space  -------------------------------------------

            meta_model = l2l.algorithms.MAML(self.critic1[index], lr=self.critic_lr)  # 元模型
            meta_opt = torch.optim.Adam(meta_model.parameters(), lr=self.critic_lr)
            learner = meta_model.clone()
            # compute td error
            with torch.no_grad():
                next_actions_t, next_log_probs_t = self.actforward(next_obss_t, index)
                next_q_t = torch.min(
                    self.critic1_old[index](next_obss_t, next_actions_t),
                    self.critic2_old[index](next_obss_t, next_actions_t)
                )
                if not self._deterministic_backup:
                    next_q_t -= self._alpha * next_log_probs_t

            target_q_t = rewards_t + self._gamma * (1 - terminals_t) * next_q_t
            q1_t = learner(obss_t, actions_t)
            critic1_loss_t = ((q1_t - target_q_t).pow(2)).mean()
            # compute conservative loss
            random_actions_t = torch.FloatTensor(
                    batch_size_t * self._num_repeat_actions, actions_t.shape[-1]
                ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor[index].device)
            # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
            tmp_obss_t = obss_t.unsqueeze(1) \
                .repeat(1, self._num_repeat_actions, 1) \
                .view(batch_size_t * self._num_repeat_actions, obss_t.shape[-1])
            tmp_next_obss_t = next_obss_t.unsqueeze(1) \
                .repeat(1, self._num_repeat_actions, 1) \
                .view(batch_size_t * self._num_repeat_actions, obss_t.shape[-1])

            obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss_t, tmp_obss_t, learner, self.critic2[index], index)
            next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss_t, tmp_obss_t, learner, self.critic2[index], index)
            random_value1, random_value2 = self.calc_random_values(tmp_obss_t, random_actions_t, learner, self.critic2[index])

            for value in [
                obs_pi_value1, obs_pi_value2, next_obs_pi_value1, next_obs_pi_value2,
                random_value1, random_value2
            ]:
                value.reshape(batch_size_t, self._num_repeat_actions, 1)

            # cat_q shape: (batch_size, 3 * num_repeat, 1)
            cat_q1_t = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)

            conservative_loss1_t = \
                torch.logsumexp(cat_q1_t / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
                q1_t.mean() * self._cql_weight

            critic1_loss_t = critic1_loss_t + conservative_loss1_t
            learner.adapt(critic1_loss_t)
            #--------------------------------- end -----------------------------------------

            # compute td error
            with torch.no_grad():
                next_actions_t, next_log_probs_t = self.actforward(next_obss_t, index)
                next_q_t = torch.min(
                    self.critic1_old[index](next_obss_t, next_actions_t),
                    self.critic2_old[index](next_obss_t, next_actions_t)
                )
                if not self._deterministic_backup:
                    next_q_t -= self._alpha * next_log_probs_t

            target_q_t = rewards_t + self._gamma * (1 - terminals_t) * next_q_t
            q1_t = meta_model(obss_t, actions_t)
            critic1_loss_t = ((q1_t - target_q_t).pow(2)).mean()
            # compute conservative loss
            random_actions_t = torch.FloatTensor(
                batch_size_t * self._num_repeat_actions, actions_t.shape[-1]
            ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor[index].device)
            # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
            tmp_obss_t = obss_t.unsqueeze(1) \
                .repeat(1, self._num_repeat_actions, 1) \
                .view(batch_size_t * self._num_repeat_actions, obss_t.shape[-1])
            tmp_next_obss_t = next_obss_t.unsqueeze(1) \
                .repeat(1, self._num_repeat_actions, 1) \
                .view(batch_size_t * self._num_repeat_actions, obss_t.shape[-1])

            obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss_t, tmp_obss_t, meta_model, self.critic2[index], index)
            next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss_t, tmp_obss_t, meta_model,self.critic2[index], index)
            random_value1, random_value2 = self.calc_random_values(tmp_obss_t, random_actions_t, meta_model, self.critic2[index])

            for value in [
                obs_pi_value1, obs_pi_value2, next_obs_pi_value1, next_obs_pi_value2,
                random_value1, random_value2
            ]:
                value.reshape(batch_size_t, self._num_repeat_actions, 1)

            # cat_q shape: (batch_size, 3 * num_repeat, 1)
            cat_q1_t = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)

            conservative_loss1_t = \
                torch.logsumexp(cat_q1_t / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
                q1_t.mean() * self._cql_weight

            critic1_loss_t = critic1_loss_t + conservative_loss1_t

            # ----------------------------------   2 -------------------------------------
            # compute td error
            with torch.no_grad():
                next_actions, next_log_probs = self.actforward(next_obss, index)
                next_q = torch.min(
                    self.critic1_old[index](next_obss, next_actions),
                    self.critic2_old[index](next_obss, next_actions)
                )
                if not self._deterministic_backup:
                    next_q -= self._alpha * next_log_probs

            std_Q_critic_list = (self.get_variance(next_obss,next_actions,1))**(2)
            std_Q_critic_list1 = torch.where(std_Q_critic_list.double() < 1.0, 1.0, std_Q_critic_list.double())
            std_Q_critic_list1 = std_Q_critic_list1**(-1)
            target_q = rewards + self._gamma * (1 - terminals) * next_q
            q1, q2 = learner(obss, actions), self.critic2[index](obss, actions)

            # tuber loss
            if ((q1 - target_q).pow(2)).mean()>100:
                critic1_loss = torch.mul(std_Q_critic_list1, 10* (abs(q1 - target_q) - 5))
                critic1_loss = (critic1_loss).mean()
            else:
                critic1_loss = torch.mul(std_Q_critic_list1,(q1 - target_q).pow(2))
                critic1_loss = (critic1_loss).mean()
            critic2_loss = ((q2 - target_q).pow(2)).mean()

            # compute conservative loss
            random_actions = torch.FloatTensor(
                batch_size * self._num_repeat_actions, actions.shape[-1]
            ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor[index].device)
            # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
            tmp_obss = obss.unsqueeze(1) \
                .repeat(1, self._num_repeat_actions, 1) \
                .view(batch_size * self._num_repeat_actions, obss.shape[-1])
            tmp_next_obss = next_obss.unsqueeze(1) \
                .repeat(1, self._num_repeat_actions, 1) \
                .view(batch_size * self._num_repeat_actions, obss.shape[-1])

            obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss, tmp_obss, learner, self.critic2[index], index)
            next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss, tmp_obss, learner, self.critic2[index], index)
            random_value1, random_value2 = self.calc_random_values(tmp_obss, random_actions, learner, self.critic2[index])

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

            critic1_loss = critic1_loss + conservative_loss1 + critic1_loss_t

            meta_opt.zero_grad()
            critic1_loss.backward(retain_graph=True)
            meta_opt.step()

            critic2_loss = critic2_loss + conservative_loss2

            self.critic2_optim[index].zero_grad()
            critic2_loss.backward()
            self.critic2_optim[index].step()

            self._sync_weight(index)

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        return result
    
    # 计算损失和梯度，但是不执行更新操作
    def learn_data_split(self, batch: Dict) :
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        batch_size = obss.shape[0]

        # compute td error
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss, 0)
            next_q = torch.min(
                self.critic1_old[0](next_obss, next_actions),
                self.critic2_old[0](next_obss, next_actions)
            )
            if not self._deterministic_backup:
                next_q -= self._alpha * next_log_probs

        target_q = rewards + self._gamma * (1 - terminals) * next_q
        q1 = self.critic1[0](obss, actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        # compute conservative loss
        random_actions = torch.FloatTensor(
            batch_size * self._num_repeat_actions, actions.shape[-1]
        ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor[0].device)
        # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
        tmp_obss = obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        tmp_next_obss = next_obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        
        obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss, tmp_obss, self.critic1[0],self.critic2[0],0)
        next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss, tmp_obss, self.critic1[0], self.critic2[0],0)
        random_value1, random_value2 = self.calc_random_values(tmp_obss, random_actions, self.critic1[0],self.critic2[0])

        for value in [
            obs_pi_value1, obs_pi_value2, next_obs_pi_value1, next_obs_pi_value2,
            random_value1, random_value2
        ]:
            value.reshape(batch_size, self._num_repeat_actions, 1)
        
        # cat_q shape: (batch_size, 3 * num_repeat, 1)
        cat_q1 = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)

        conservative_loss1 = \
            torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q1.mean() * self._cql_weight
        
        critic2_loss = critic1_loss + conservative_loss1
        for m in self.critic1[0].parameters():
            if m.grad is not None:
                m.grad.detach_()
                m.grad.zero_()
        grad = torch.autograd.grad(critic2_loss, self.critic1[0].parameters())
        return {
            "loss": critic2_loss,
            "grad": grad
        }

    def get_variance(self, obs, acts, update_type = 1):
        std_Q_list1, std_Q_list2 = [], []
        L_target_Q1, L_target_Q2 = [], []
        for index in range(self.num_ensemble):
            if update_type == 0:
                target_Q1 = self.critic1[index](obs, acts)
                target_Q2 = self.critic2[index](obs, acts)
            else:
                target_Q1 = self.critic1_old[index](obs, acts)
                target_Q2 = self.critic2_old[index](obs, acts)
            L_target_Q1.append(target_Q1)
            L_target_Q2.append(target_Q2)

        std_Q_list1.append(torch.stack(L_target_Q1).std(axis=0).detach())
        return torch.stack(std_Q_list1).squeeze(0)

    def learnCopy(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
                                                       batch["next_observations"], batch["rewards"], batch["terminals"]
        batch_size = obss.shape[0]

        for index in range(self.num_ensemble):

            # update actor
            a, log_probs = self.actforward(obss, index)
            q1a, q2a = self.critic1[index](obss, a), self.critic2[index](obss, a)
            actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()
            self.actor_optim[index].zero_grad()
            actor_loss.backward()
            self.actor_optim[index].step()

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
                    next_actions, next_log_probs = self.actforward(next_obss, index)
                    next_q = torch.min(
                        self.critic1_old[index](next_obss, next_actions),
                        self.critic2_old[index](next_obss, next_actions)
                    )
                    if not self._deterministic_backup:
                        next_q -= self._alpha * next_log_probs

            target_q = rewards + self._gamma * (1 - terminals) * next_q
            q1, q2 = self.critic1[index](obss, actions), self.critic2[index](obss, actions)

            std_Q_critic_list = (self.get_variance(next_obss, next_actions, 1)) ** (2)
            std_Q_critic_list1 = torch.where(std_Q_critic_list.double() < 1.0, 1.0, std_Q_critic_list.double())
            std_Q_critic_list1 = std_Q_critic_list1 ** (-1)

            critic1_loss = torch.mul(std_Q_critic_list1, (q1 - target_q).pow(2))
            critic1_loss = (critic1_loss).mean()
            critic2_loss = ((q2 - target_q).pow(2)).mean()


            # compute conservative loss
            random_actions = torch.FloatTensor(
                batch_size * self._num_repeat_actions, actions.shape[-1]
            ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor[index].device)
            # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
            tmp_obss = obss.unsqueeze(1) \
                .repeat(1, self._num_repeat_actions, 1) \
                .view(batch_size * self._num_repeat_actions, obss.shape[-1])
            tmp_next_obss = next_obss.unsqueeze(1) \
                .repeat(1, self._num_repeat_actions, 1) \
                .view(batch_size * self._num_repeat_actions, obss.shape[-1])

            obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss, tmp_obss, self.critic1[index], self.critic2[index], index)
            next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss, tmp_obss, self.critic1[index], self.critic2[index], index)
            random_value1, random_value2 = self.calc_random_values(tmp_obss, random_actions,self.critic1[index], self.critic2[index])

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

            if self._with_lagrange:
                cql_alpha = torch.clamp(self.cql_log_alpha.exp(), 0.0, 1e6)
                conservative_loss1 = cql_alpha * (conservative_loss1 - self._lagrange_threshold)
                conservative_loss2 = cql_alpha * (conservative_loss2 - self._lagrange_threshold)

                self.cql_alpha_optim.zero_grad()
                cql_alpha_loss = -(conservative_loss1 + conservative_loss2) * 0.5
                cql_alpha_loss.backward(retain_graph=True)
                self.cql_alpha_optim.step()

            critic1_loss = critic1_loss + conservative_loss1
            critic2_loss = critic2_loss + conservative_loss2

            # update critic
            self.critic1_optim[index].zero_grad()
            critic1_loss.backward(retain_graph=True)
            self.critic1_optim[index].step()

            self.critic2_optim[index].zero_grad()
            critic2_loss.backward()
            self.critic2_optim[index].step()

            self._sync_weight(index)

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        if self._with_lagrange:
            result["loss/cql_alpha"] = cql_alpha_loss.item()
            result["cql_alpha"] = cql_alpha.item()

        return result
