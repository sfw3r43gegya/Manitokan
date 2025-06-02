import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class QMixer(nn.Module):
    """
    Computes total Q values given agent q values and global states.
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param num_agents: (int) number of agents in env
    :param cent_obs_dim: (int) dimension of the centralized state
    :param device: (torch.Device) torch device on which to do computations.
    :param multidiscrete_list: (list) list of each action dimension if action space is multidiscrete
    """

    def __init__(self, args, num_agents, cent_obs_dim, device, multidiscrete_list=None, ppo=False):
        """
        init mixer class
        """
        super(QMixer, self).__init__()
        self.device = device
        self.num_agents = num_agents
        self.cent_obs_dim = cent_obs_dim
        self.use_orthogonal = args.use_orthogonal
        self.name= "qmix"

        # dimension of the hidden layer of the mixing net
        self.hidden_layer_dim = args.mixer_hidden_dim
        # dimension of the hidden layer of each hypernet
        self.hypernet_hidden_dim = args.hypernet_hidden_dim

        if multidiscrete_list and not ppo:
            self.num_mixer_q_inps = sum(multidiscrete_list)
        elif not ppo:
            self.num_mixer_q_inps = self.num_agents
        else:
            self.num_mixer_q_inps = 1

        if self.use_orthogonal:
            def init_(m):
                return init(m, nn.init.orthogonal_,
                            lambda x: nn.init.constant_(x, 0))
        else:
            def init_(m):
                return init(m, nn.init.xavier_uniform_,
                            lambda x: nn.init.constant_(x, 0))

        # hypernets output the weight and bias for the 2 layer MLP which takes in the state and agent Qs and outputs Q_tot

        modules_1 = []
        modules_2 = []
        modules_1.append(init_(nn.Linear(
            self.cent_obs_dim, self.num_mixer_q_inps * self.hidden_layer_dim)).to(self.device))

        modules_2.append(init_(nn.Linear(self.cent_obs_dim, self.hidden_layer_dim)).to(self.device))

        for i in range(args.hypernet_layers):


            if i > 0:
                modules_1.append(nn.ReLU())
                modules_1.append(init_(nn.Linear(self.hypernet_hidden_dim,
                                self.num_mixer_q_inps * self.hidden_layer_dim)))

                modules_2.append(nn.ReLU())
                modules_2.append( init_(nn.Linear(self.hypernet_hidden_dim, self.hidden_layer_dim)))

        self.hyper_w1 = nn.Sequential(*modules_1).to(self.device)
        self.hyper_w2 = nn.Sequential(*modules_2).to(self.device)


        # hyper_b1 outputs bias vector of dimension (1 x hidden_layer_dim)
        self.hyper_b1 = init_(
            nn.Linear(self.cent_obs_dim, self.hidden_layer_dim)).to(self.device)
        # hyper_b2 outptus bias vector of dimension (1 x 1)
        self.hyper_b2 = nn.Sequential(
            init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(self.hypernet_hidden_dim, 1))
        ).to(self.device)

    def forward(self, agent_q_inps, states=None):
        """
         Computes Q_tot using the individual agent q values and global state.
         :param agent_q_inps: (torch.Tensor) individual agent q values
         :param states: (torch.Tensor) state input to the hypernetworks.
         :return Q_tot: (torch.Tensor) computed Q_tot values
         """
        batch_size = agent_q_inps.size(0)
        # states : (batch_size * T, state_dim)
        states = states.reshape(-1, self.cent_obs_dim)
        # agent_qs: (batch_size * T, 1, num_agents)
        agent_qs = agent_q_inps.reshape(-1, agent_q_inps.size(3), self.num_agents)

        # First layer w and b
        w1 = torch.abs(self.hyper_w1(states))
        # w1: (batch_size * T, num_agents, embed_dim)
        w1 = w1.view(-1, self.num_agents, self.hidden_layer_dim)
        b1 = self.hyper_b1(states)
        b1 = b1.view(-1, 1, self.hidden_layer_dim)

        # Second layer w and b
        # w2 : (batch_size * T, embed_dim)
        w2 = torch.abs(self.hyper_w2(states))
        # w2 : (batch_size * T, embed_dim, 1)
        w2 = w2.view(-1, self.hidden_layer_dim, 1)
        # State-dependent bias
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        # First hidden layer
        # hidden: (batch_size * T,  1, embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Compute final output
        # y: (batch_size * T,  1, 1)
        y = torch.bmm(hidden, w2) + b2
        # Reshape and return
        # q_total: (batch_size, T, 1)
        q_total = y.view(batch_size, -1, agent_q_inps.size(3),1)
        return q_total.repeat(1,1,1,self.num_agents)