from pdb import set_trace as T

import gymnasium
import functools

import pufferlib.emulation
import pufferlib.environments
import pufferlib.postprocess

import src_minigrid

ALIASES = {
    'minigrid': 'MiniGrid-LavaGapS7-v0',
}


def env_creator(name='minigrid'):
    return functools.partial(make, name=name)

def make(env, render_mode='rgb_array', buf=None):
   # if name in ALIASES:
      #  name = ALIASES[name]

   # minigrid = pufferlib.environments.try_import('minigrid')
   # env = gymnasium.make(name, render_mode=render_mode)
    env = MiniGridWrapper(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return src_minigrid.envs.PufferLib_2.pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

class MiniGridWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space =  self.env.observation_space

        self.action_space = self.env.action_space
        self.close = self.env.close
        self.render = self.env.render
        self.close = self.env.close
        self.render_mode = 'rgb_array'

    def reset(self, seed=None, options=None):
        self.tick = 0
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)




        return obs, reward, done, truncated, info
