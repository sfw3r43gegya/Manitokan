import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QTranBase(nn.Module):
    def __init__(self, args,cent_obs_dim, num_agents, device, ppo=False):
        super(QTranBase, self).__init__()
        self.name = "qtrans"
        self.args = args

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = args.obs_shape*2
        self.arch = self.args.qtran_arch # QTran architecture

        self.embed_dim = args.mixer_hidden_dim

        # Q(s,u)
        if self.arch == "coma_critic":
            # Q takes [state, u] as input
            q_input_size = self.state_dim + (self.n_agents * self.n_actions)
        elif self.arch == "qtran_paper":
            # Q takes [state, agent_action_observation_encodings]
            q_input_size = self.state_dim + self.args.rnn_hidden_dim + self.n_actions
        else:
            raise Exception("{} is not a valid QTran architecture".format(self.arch))
            
        if self.args.network_size == "small":
            self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, 1, device=args.device))

            # V(s)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, 1, device=args.device))
            ae_input = self.args.rnn_hidden_dim + self.n_actions
            self.action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input, device=args.device),
                                                 nn.GELU(),
                                                 nn.Linear(ae_input, ae_input, device=args.device))
        elif self.args.network_size == "big":
            self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, 1, device=args.device))
            # V(s)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, 1, device=args.device))
            ae_input = self.args.rnn_hidden_dim + self.n_actions
            self.action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input, device=args.device),
                                                 nn.GELU(),
                                                 nn.Linear(ae_input, ae_input, device=args.device))
        else:
            assert False

    def forward(self, batch, hidden_states, batch_size, max_seq_length,  actions=None, ent = None):
        bs = batch_size
        ts = max_seq_length

        states = batch["obs"].reshape(bs * ts, self.state_dim) if ent is None else batch["obs"].reshape(bs * ts, self.state_dim)

        if self.arch == "coma_critic":
            if actions is None:
                # Use the actions taken by the agents
                actions = th.from_numpy(np.eye(self.n_actions)[batch["actions"]].reshape(bs * ts, self.n_agents * self.n_actions)).to(th.float32)
            else:
                # It will arrive as (bs, ts, agents, actions), we need to reshape it
                actions = actions.reshape(bs * ts, self.n_agents * self.n_actions)
            inputs = th.cat([states, actions], dim=1)
        elif self.arch == "qtran_paper":
            if actions is None:
                # Use the actions taken by the agents
                actions = th.from_numpy(np.eye(self.n_actions)[batch["actions"].cpu().numpy()].reshape(bs * ts, self.n_agents, self.n_actions)).to(th.float32)
            else:
                # It will arrive as (bs, ts, agents, actions), we need to reshape it
                actions = actions.reshape(bs * ts, self.n_agents, self.n_actions)

            hidden_states = hidden_states.reshape(bs * ts, self.n_agents, -1)
            agent_state_action_input = th.cat([hidden_states.to(self.args.device), actions.to(self.args.device)], dim=2)
            if ent is None:
                agent_state_action_encoding = self.action_encoding(agent_state_action_input.reshape(bs * ts * self.n_agents, -1)).reshape(bs * ts, self.n_agents, -1)
                agent_state_action_encoding = agent_state_action_encoding.sum(dim=1) # Sum across agents
                inputs = th.cat([states, agent_state_action_encoding], dim=1)

        q_outputs = self.Q(inputs)

        states = batch["obs"].reshape(bs * ts, self.state_dim)
        v_outputs = self.V(states)

        return q_outputs.repeat(1,self.n_agents), v_outputs.repeat(1,self.n_agents)


class QTranAlt(nn.Module):
    def __init__(self, args,cent_obs_dim, num_agents, device, ppo=False):
        super(QTranAlt, self).__init__()

        self.args = args
        self.name = "qtrans"
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        # Q(s,-,u-i)
        # Q takes [state, u-i, i] as input
        q_input_size = self.state_dim + (self.n_agents * self.n_actions) + self.n_agents

        if self.args.network_size == "small":
            self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, self.n_actions, device=args.device))

            # V(s)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, 1, device=args.device))
        elif self.args.network_size == "big":
             # Adding another layer
             self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, self.n_actions, device=args.device))
            # V(s)
             self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, self.embed_dim, device=args.device),
                                   nn.GELU(),
                                   nn.Linear(self.embed_dim, 1, device=args.device))
        else:
            assert False

    def forward(self, batch, states= None, masked_actions=None):
        bs = batch
        ts = batch.max_seq_length
        # Repeat each state n_agents times
        repeated_states = batch["obs"].repeat(1, 1, self.n_agents).view(-1, self.state_dim)

        if masked_actions is None:
            actions = batch["actions"].repeat(1, 1, self.n_agents, 1)
            agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
            agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions)#.view(self.n_agents, -1)
            masked_actions = actions * agent_mask.unsqueeze(0).unsqueeze(0)
            masked_actions = masked_actions.view(-1, self.n_agents * self.n_actions)

        agent_ids = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).repeat(bs, ts, 1, 1).view(-1, self.n_agents)

        inputs = th.cat([repeated_states, masked_actions, agent_ids], dim=1)

        q_outputs = self.Q(inputs)

        states = batch["state"].repeat(1,1,self.n_agents).view(-1, self.state_dim)
        v_outputs = self.V(states)

        return q_outputs, v_outputs