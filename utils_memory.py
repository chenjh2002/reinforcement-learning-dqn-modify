from typing import (
    Tuple,
)

import torch
from torch.utils.data import Dataset

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)


class ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
            full_sink: bool = True,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 0

        sink = lambda x: x.to(device) if full_sink else x
        self.m_states = sink(torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8))
        self.m_actions = sink(torch.zeros((capacity, 1), dtype=torch.long))
        self.m_rewards = sink(torch.zeros((capacity, 1), dtype=torch.int8))
        self.m_dones = sink(torch.zeros((capacity, 1), dtype=torch.bool))

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:
        self.m_states[self.__pos] = folded_state
        self.m_actions[self.__pos, 0] = action
        self.m_rewards[self.__pos, 0] = reward
        self.m_dones[self.__pos, 0] = done

        self.__pos += 1
        self.__size = max(self.__size, self.__pos)
        self.__pos %= self.__capacity

    def sample(self, batch_size: int) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        indices = torch.randint(0, high=self.__size, size=(batch_size,))
        b_state = self.m_states[indices, :4].to(self.__device).float()
        b_next = self.m_states[indices, 1:].to(self.__device).float()
        b_action = self.m_actions[indices].to(self.__device)
        b_reward = self.m_rewards[indices].to(self.__device).float()
        b_done = self.m_dones[indices].to(self.__device).float()
        return b_state, b_action, b_reward, b_next, b_done

    def __len__(self) -> int:
        return self.__size
    
    def __getitem__(self,index):
        b_state = self.m_states[index, :4].to(self.__device).float()
        b_next = self.m_states[index, 1:].to(self.__device).float()
        b_action = self.m_actions[index].to(self.__device)
        b_reward = self.m_rewards[index].to(self.__device).float()
        b_done = self.m_dones[index].to(self.__device).float()
        return b_state, b_action, b_reward, b_next, b_done
