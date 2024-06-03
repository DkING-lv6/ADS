import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy

import random

# model-free policy trainer
class MFPolicyDataSplitTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        dataSet: dict,
        buffer: ReplayBuffer,
        logger: Logger,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        mini_batch_size: int = 128,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        file_name: str = "dc",
        step_per_split: int = 250,
        sample_scale: int = 200,
        split_ratio: float = 0.5
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.dataSet = dataSet
        self.buffer = buffer
        self.logger = logger

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._mini_batch_size = mini_batch_size
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler
        self.file_name = file_name
        self.step_per_split = step_per_split
        self.sample_scale = sample_scale
        self.split_ratio = split_ratio

        rewards = dataSet["rewards"]
        self.S_T = []
        self.S_V = []
        sortl = sorted(range(len(rewards)), key=lambda k: rewards[k])
        for i in range(10):
            itemList = sortl[i * 100000: (1 + i) * 100000]
            modIndex = int(len(itemList) * 0.5)
            self.S_T.append(itemList[0:modIndex])
            self.S_V.append(itemList[modIndex:len(itemList)])

    def train_data_split(self) -> Dict[str, List[int]]:
        start_time = time.time()
        num_timesteps = 0
        last_10_performance = deque(maxlen=10)

        for e in range(1, self._epoch + 1):

            self.policy.train()
            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")

            if e < 300:
                for it in pbar:
                    batch_v = self.buffer.sample(260)

                    # batch_t = self.buffer.sample(int(self._batch_size * 0.8))
                    # batch_v = self.buffer.sample(self._batch_size - int(self._batch_size * 0.8))

                    loss = self.policy.learnCopy(batch_v)
                    pbar.set_postfix(**loss)

                    for k, v in loss.items():
                        self.logger.logkv_mean(k, v)
                            # ''''''
                    num_timesteps += 1
                    # ''''''
            else:
                counts = 0
                for it in pbar:
                    batch_t,batch_v = [],[]
                    for i in range(10):
                        batch_t.extend(random.sample(self.S_T[i], int(self._batch_size * self.split_ratio)))
                        batch_v.extend(random.sample(self.S_V[i], self._batch_size - int(self._batch_size * self.split_ratio)))
                    batch = batch_t + batch_v
                    batch_t = self.buffer.sample_index(batch_t)
                    batch_v = self.buffer.sample_index(batch_v)
                    batch_all = self.buffer.sample_index(batch)

                    # batch_indexes = np.random.randint(0, len(self.dataSet["rewards"]), size=self._batch_size)
                    # batch_t = self.buffer.sample_index(batch_indexes[0:int(self._batch_size * 0.8)])
                    # batch_v = self.buffer.sample_index(batch_indexes[int(self._batch_size * 0.8):self._batch_size])
                    # batch_all = self.buffer.sample_index(batch_indexes)

                    # replay.append(batch_all)
                    loss = self.policy.learn(batch_v, batch_t, batch_all)
                    pbar.set_postfix(**loss)

                    for k, v in loss.items():
                        self.logger.logkv_mean(k, v)
                    num_timesteps += 1
                    counts +=1

                    # if counts % 200 == 0:
                    #     for item in replay:
                    #         loss = self.policy.learn_policy(item)
                    #         pbar.set_postfix(**loss)
                    #         for k, v in loss.items():
                    #             self.logger.logkv_mean(k, v)
                    #     replay = []

                # self.update_subsets()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # evaluate current policy
            eval_info = self._evaluate()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
            norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
            last_10_performance.append(norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs()

            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))
        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = self.policy.select_action(obs.reshape(1,-1), deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())

            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }

    def update_subsets(self) :
        splitList = []
        for i in range(10):
            S_T_adds = random.sample(self.S_T[i], self.sample_scale)
            for j in S_T_adds:
                self.S_T[i].remove(j)
            S_V_adds = random.sample(self.S_V[i], self.sample_scale)
            for k in S_V_adds:
                self.S_V[i].remove(k)
            adds = S_T_adds + S_V_adds
            splitList.append(adds)

        for i in range(10):
            alpha = 1e-6
            losses = []
            gradients = []
            batch_index = splitList[i]
            for batch_i in batch_index:
                batch = self.buffer.sample_index([batch_i])
                loss = self.policy.learn_data_split1(batch)
                gradList = []
                for item in loss.get("grad"):
                    gradList.append(torch.flatten(item))
                losses.append(loss.get("loss"))
                gradients.append(torch.cat(gradList,0))

            A = random.sample(gradients,1)
            A = A[0]
            S_v_add = []
            S_t_add = []
            for j in range(3):
                adjusted_losses = [ls - alpha * torch.dot(gr,A) for ls,gr in zip(losses, gradients)]
                _, indices = torch.topk(torch.tensor(adjusted_losses), self.sample_scale)
                S_v_indices = indices.tolist()
                S_v_add = [batch_index[k] for k in S_v_indices]
                S_t_add = list(set(batch_index).difference(S_v_add))
                S_t_indices = list(set(range(len(batch_index))) - set(S_v_indices))
                A = torch.mean(torch.stack([gradients[k] for k in S_t_indices]), dim=0)
            self.S_T[i].extend(S_t_add)
            self.S_V[i].extend(S_v_add)