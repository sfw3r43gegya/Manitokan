REGISTRY = {}

from .basic_controller import BasicMAC
from .distributed_controller import DistributedMAC
from.noise_controller import NoiseMAC
from learners.centralized.PPO_Learners.PPO_Policies.MappoPolicy import R_MAPPOPolicy

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["distributed_mac"] = DistributedMAC
REGISTRY["mappo_policy"] = R_MAPPOPolicy
REGISTRY["maven_policy"] = NoiseMAC
