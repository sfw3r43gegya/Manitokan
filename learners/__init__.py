# https://github.com/jshe/agent-time-attention/tree/main
from .coma_learner import COMALearner
from .pg_learner import PGLearner
from .centralized.PPO_Learners.mappo_learner import MAPPOLearner

from .centralized.ValueDecomp.qmix_learner import QMixAgent
from .centralized.ValueDecomp.maven_learner import QLearner
from .centralized.stateful_active_facilitator.saf_learner import SAFLearner
from .random_agent import RandomAgent



REGISTRY = {}

REGISTRY["coma_learner"] = COMALearner
REGISTRY["pg_learner"] = PGLearner
REGISTRY["mappo_learner"] = MAPPOLearner

REGISTRY["qmixer_learner"] = QMixAgent
REGISTRY["maven_learner"] = QLearner
REGISTRY["saf_learner"] = SAFLearner
REGISTRY["random_agent"] = RandomAgent


