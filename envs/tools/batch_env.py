# Lint as: python2, python3
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Threaded batch environment wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from concurrent import futures

from six.moves import range
from six.moves import zip

import envs.tools.nest_utils as nest_utils
import scipy.spatial.distance as dist
from numpy import random
import numpy as np
from envs.PufferLib_2.pufferlib import emulation


class BatchEnv(object):
  """Wrapper that steps multiple environments in separate threads.

  The threads are stepped in lock step, so all threads progress by one step
  before any move to the next step.
  """

  def __init__(self, batch_size, puffer, env_builder, **env_kwargs):
    self.batch_size = batch_size
    random.seed(np.int64(env_kwargs["seed"]))
    self.env_seeds = [random.randint(1, 9000) for _ in range(batch_size)]
    self.puffer = puffer
    envs = []

    for seed in self.env_seeds:
      env_kwargs["seed"] = seed

      if "layout_name" in env_kwargs and env_kwargs["layout_name"] == "forced_coordination":
        envs.append(env_builder(layout_name= "forced_coordination"))

      elif "map_name" in env_kwargs:
        envs.append(env_builder(map_name=env_kwargs["map_name"],
                                seed=seed,
                                window_size_x=1920,
                                window_size_y=1200))

      else:
        env_kwargs.pop("layout_name", None)

        if puffer:

          def mano():
            env_init = env_builder

            return emulation.GymnasiumPufferEnv(env_creator=env_init(**env_kwargs))

          envs.append(mano())

        else:
          envs.append(env_builder(**env_kwargs))


    self._envs = envs
    self._executor = futures.ThreadPoolExecutor(max_workers=self.batch_size)
    if "layout_name" in env_kwargs and env_kwargs["layout_name"] == "forced_coordination":
      self._num_actions = envs[0].action_space
      self._n_agents = 2
      self._observation_shape = envs[0].observation_space
      self._episode_length = env_kwargs["horizon"]
      self.name = "forced_coordination"


    elif "map_name" in env_kwargs:
      env_info = self._envs[0].get_env_info()
      self.name = "starcraft"
      self._num_actions = env_info["n_actions"]
      reset_list = []

      def init_sc(env):

          return env.reset()


      
      for env in self._envs:
        reset_list.append(self._executor.submit(init_sc, env))



      reset_list = [env_output.result() for env_output in reset_list]

      self._observation_shape =  self._envs[0].get_obs()[0].shape#*self._envs[0].get_state()[0].shape
      self._episode_length = int(env_info["episode_limit"])
      self._n_agents =  env_info["n_agents"]


    else:
      self.name = "iw2si"
      self._num_actions = self._envs[0].action_space.n
      self._observation_shape = self._envs[0].observation_space.shape
      self._episode_length = int(env_kwargs["episode_limit"])
      self._n_agents = None if not hasattr(self._envs[0], "n_agents") else self._envs[0].n_agents
      self._n_keys = None if not hasattr(self._envs[0], "n_keys") else self._envs[0].n_keys



  def reset(self):
    """Reset the entire batch of environments."""

    def reset_environment(env):

      if self.name == "iw2si":
        return env.reset()

      elif self.name == "starcraft":

        return env.reset()

      else:
        return env.multi_reset()

    try:
      output_list = []

      
      for env in self._envs:
        output_list.append(self._executor.submit(reset_environment, env))



      output_list = [env_output.result() for env_output in output_list]

    except KeyboardInterrupt:
      self._executor.shutdown(wait=True)
      raise


    observations = nest_utils.nest_stack(output_list) if self.name == "iw2si" else None
    return observations

  def step(self, action_list, agent_order):
    """Step batch of envs.

    Args:
      action_list: A list of actions, one per environment in the batch. Each one
        should be a scalar int or a numpy scaler int.

    Returns:
      A tuple (observations, rewards):
        observations: A nest of observations, each one a numpy array where the
          first dimension has size equal to the number of environments in the
          batch.
        rewards: An array of rewards with size equal to the number of
          environments in the batch.
    """

    def step_environment(tup):

      if self.name == "iw2si":
        return tup[0].step(tup[1], tup[2] )

      elif self.name == "starcraft":
        return  tup[0].step(tup[1])

      else:
        return env.multi_step(tup[1][0], tup[1][1] )

    try:
      output_list = []


      for env, action, agent_order in zip(self._envs, action_list, agent_order):

          output_list.append(self._executor.submit(step_environment, (env, action, agent_order)))



      output_list = [env_output.result()[:2] for env_output in output_list]
   
    except KeyboardInterrupt:
      self._executor.shutdown(wait=True)
      raise

    if self.name == "iw2si":
      rewards, dones = nest_utils.nest_stack(output_list)

      return rewards, dones

    elif self.name == "starcraft":

      rewards, dones = nest_utils.nest_stack(output_list)
      return rewards, dones

    elif self.name == "forced_coordination":
      obs, rewards, dones, infos = nest_utils.nest_stack(output_list)
      return obs, rewards, dones, infos

  def get_obs(self):

    def get_obs_environment(env):
      return env.get_obs()

    try:
      output_list = []


      for env in self._envs:
        output_list.append(self._executor.submit(get_obs_environment, env))



      output_list = [env_output.result(timeout=800) for env_output in output_list]

    except KeyboardInterrupt:
      self._executor.shutdown(wait=True)
      raise

    return nest_utils.nest_stack(output_list)


  def get_avail_actions(self):

    def get_available_actions_environment(env):

      if self.name == "starcraft":
        avail_actions = []
        for agent_id in range(self._n_agents):
          avail_actions.append( env.get_avail_agent_actions(agent_id))

        return nest_utils.nest_stack(avail_actions)

      else:
        return env.get_avail_actions()

    try:
      output_list = []


      for env in self._envs:
        output_list.append(self._executor.submit(get_available_actions_environment, env))



      output_list = [env_output.result() for env_output in output_list]

    except KeyboardInterrupt:
      self._executor.shutdown(wait=True)
      raise

    return output_list


  def get_state(self):

    def get_state_environment(env):
      return env.get_state()

    try:
      output_list = []


      for env in self._envs:
        output_list.append(self._executor.submit(get_state_environment, env))



      output_list = [env_output.result() for env_output in output_list]

    except KeyboardInterrupt:
      self._executor.shutdown(wait=True)
      raise

    return nest_utils.nest_stack(output_list)


  def get_carryings(self):

    def get_carryings_environment(env):

      if self.puffer:
        return env.get_carryings()
      else:
        return env.carryings

    try:
      output_list = []


      for env in self._envs:
        output_list.append(self._executor.submit(get_carryings_environment, env))



      output_list = [env_output.result() for env_output in output_list]

    except KeyboardInterrupt:
      self._executor.shutdown(wait=True)
      raise

    return output_list


  def open_doors(self):

    def open_doors_environment(env):
      return env.doors_opened

    try:
      output_list = []


      for env in self._envs:
        output_list.append(self._executor.submit(open_doors_environment, env))



      output_list = [env_output.result() for env_output in output_list]

    except KeyboardInterrupt:
      self._executor.shutdown(wait=True)
      raise

    return output_list


  def euc_dist(self):

    def get_euc_dist_environment(env):

      return dist.euclidean(env.agent_posits()[0], env.agent_posits()[1])

    try:
      output_list = []


      for env in self._envs:
        output_list.append(self._executor.submit(get_euc_dist_environment, env))



      output_list = [env_output.result() for env_output in output_list]

    except KeyboardInterrupt:
      self._executor.shutdown(wait=True)
      raise

    return output_list


  def get_env_info(self):

    def get_env_info_environment(env):
      return env.get_env_info()

    try:
      output_list = []


      for env in self._envs:
        output_list.append(self._executor.submit(get_env_info_environment, env))


      output_list = [env_output.result() for env_output in output_list]

    except KeyboardInterrupt:
      self._executor.shutdown(wait=True)
      raise

    return nest_utils.nest_stack(output_list)

  @property
  def observation_shape(self):
    """Observation shape per environment, i.e. with no batch dimension."""
    return self._observation_shape

  @property
  def num_actions(self):
    return self._num_actions

  @property
  def episode_length(self):
    return int(self._episode_length)

  def last_phase_rewards(self):
    return [env.last_phase_reward() for env in self._envs]
