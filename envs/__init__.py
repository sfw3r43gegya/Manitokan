from functools import partial
from .multiagentenv import MultiAgentEnv
from .doorkey import DoorKeyEnv5x5

from .iw2si import iw2si





def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["dk"] = partial(env_fn, env=DoorKeyEnv5x5)
REGISTRY["iw2si"] = partial(env_fn, env=iw2si)

