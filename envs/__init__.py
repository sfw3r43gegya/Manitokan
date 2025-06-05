from functools import partial
from .multiagentenv import MultiAgentEnv

from .iw2si import iw2si


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}

REGISTRY["iw2si"] = partial(env_fn, env=iw2si)


