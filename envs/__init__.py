from functools import partial
from .multiagentenv import MultiAgentEnv
from .doorkey import DoorKeyEnv5x5
from .multiroom import MultiRoomEnvN2S4
from .iw2si import iw2si

from .Portal_games.env_factory import get_env
from src_minigrid.envs.SMAC.smac.env import StarCraft2Env

from src_minigrid.envs.PantheonRL.overcookedgym.overcooked import OvercookedMultiEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["dk"] = partial(env_fn, env=DoorKeyEnv5x5)
REGISTRY["iw2si"] = partial(env_fn, env=iw2si)
REGISTRY["mr"] = partial(env_fn, env=MultiRoomEnvN2S4)
REGISTRY["oc"] = OvercookedMultiEnv
REGISTRY["room30"] = get_env("room30")
REGISTRY['room30_ckpt'] = get_env('room30_ckpt')
REGISTRY['secret_room'] = get_env('secret_room')
REGISTRY['secret_room_ckpt'] = get_env('secret_room_ckpt')
REGISTRY['push_box'] = get_env('push_box')
REGISTRY['push_box_ckpt'] = get_env('push_box_ckpt')
REGISTRY['starcraft'] = StarCraft2Env #needs to pass maps string as well
