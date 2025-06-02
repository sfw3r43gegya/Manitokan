import numpy as np

import torch

from .utils import *


class ReplayBuffer:
    def __init__(self, observation_space, action_space, state_space, params, device):

        self.n_agents = params.n_agents
        self.rollout_threads = params.rollout_threads
        self.env_steps = params.env_steps
        self.continuous_action = params.continuous_action
        self.obs_shape = observation_space
        self.action_shape = action_space
        self.state_space = state_space
        self.state_shape = state_space

        self.device = device

        self.obs = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents) + self.obs_shape).float().to(
            device)

        if self.continuous_action:
            self.actions = torch.zeros(
                (self.env_steps, self.rollout_threads, self.n_agents) + self.action_shape).float().to(device)
        else:
            self.actions = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents)).float().to(device)

        self.action_masks = torch.zeros(
            (self.env_steps, self.rollout_threads, self.n_agents) + self.action_shape).float().to(device)
        self.logprobs = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents)).float().to(device)
        self.rewards = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents)).float().to(device)
        self.values = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents)).float().to(device)
        self.dones = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents)).float().to(device)
        self.state = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents) + self.state_shape).float().to(
            device)

    def insert(
            self,
            obs: torch.Tensor,
            state: torch.Tensor,
            action_masks: torch.Tensor,
            actions: torch.Tensor,
            logprobs: torch.Tensor,
            rewards: torch.Tensor,
            values: torch.Tensor,
            dones: torch.Tensor,
            step):

        self.obs[step] = obs
        self.state[step] = state

        if type(action_masks) == type(None):
            # No action masks, so make a custom action mask with ones everywhere (which means all actions are valid)
            self.action_masks[step] = torch.ones_like(self.action_masks[step])
        else:
            self.action_masks[step] = action_masks
        self.actions[step] = actions
        self.logprobs[step] = logprobs
        self.rewards[step] = rewards
        self.values[step] = values
        self.dones[step] = dones


class ReplayBufferImageObs:
    def __init__(self, observation_space, action_space, params, device):

        self.n_agents = int(params.n_agents)
        self.rollout_threads = int(params.batch_size_run)
        self.env_steps = params.episode_limit
        self.continuous_action = params.continuous_action
        if params.env_args["completion_signal"]:
            mid_shape = int(observation_space / 3)
            mid_shape_1 = int((mid_shape / 2))

            self.obs_shape = (int(mid_shape/mid_shape_1),
                              mid_shape_1, 3)
        else:
            self.obs_shape = (3, int(observation_space/9), int(observation_space/9))

        self.action_shape = (action_space,)

        self.device = device

        self.obs = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents) + self.obs_shape).float().to(
            device)

        if self.continuous_action:
            self.actions = torch.zeros(
                (self.env_steps, self.rollout_threads, self.n_agents) + self.action_shape).float().to(device)
        else:
            self.actions = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents)).float().to(device)

        self.action_masks = torch.zeros(
            (self.env_steps, self.rollout_threads, self.n_agents) + self.action_shape).float().to(device)
        self.logprobs = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents)).float().to(device)
        self.rewards = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents)).float().to(device)
        self.values = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents)).float().to(device)
        self.dones = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents)).float().to(device)

    def insert(
            self,
            obs: torch.Tensor,
            action_masks: torch.Tensor,
            actions: torch.Tensor,
            logprobs: torch.Tensor,
            rewards: torch.Tensor,
            values: torch.Tensor,
            dones: torch.Tensor,
            step
    ):

        self.obs[step] = obs

        if type(action_masks) == type(None):
            # No action masks, so make a custom action mask with ones everywhere (which means all actions are valid)
            self.action_masks[step] = torch.ones_like(self.action_masks[step])
        else:
            self.action_masks[step] = action_masks
        self.actions[step] = actions
        self.logprobs[step] = logprobs
        self.rewards[step] = rewards
        self.values[step] = values
        self.dones[step] = dones


    def instert_reward(self, rewards: torch.Tensor):
        self.rewards = rewards