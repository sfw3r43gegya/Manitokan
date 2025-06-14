import torch
import torch.nn as nn
import numpy as np


class VDNMixer(nn.Module):
    """
    Computes total Q values given agent q values and global states.
    :param args: (namespace) contains information about hyperparameters and algorithm configuration (unused).
    :param num_agents: (int) number of agents in env
    :param cent_obs_dim: (int) dimension of the centralized state (unused).
    :param device: (torch.Device) torch device on which to do computations.
    :param multidiscrete_list: (list) list of each action dimension if action space is multidiscrete
    """
    def __init__(self, args, cent_obs_dim, num_agents, device, multidiscrete_list=None, ):
        """
        init mixer class
        """
        super(VDNMixer, self).__init__()
        self.device = device
        self.num_agents = num_agents
        self.name = "vdn"

        if multidiscrete_list:
            self.num_mixer_q_inps = sum(multidiscrete_list)
        else:
            self.num_mixer_q_inps = self.num_agents

    def forward(self, agent_q_inps, states=None):
        """
        Computes Q_tot by summing individual agent q values.
        :param agent_q_inps: (torch.Tensor) individual agent q values
        :param states: (torch.Tensor) unused.

        :return Q_tot: (torch.Tensor) computed Q_tot values
        """

        if type(agent_q_inps) == np.ndarray:
            agent_q_inps = torch.FloatTensor(agent_q_inps,device=self.device)




        return agent_q_inps.sum(dim=2, keepdim=True).repeat(1,1,self.num_agents,1)