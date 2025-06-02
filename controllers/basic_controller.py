from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th



# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args, learner):
        self.n_agents = args.n_agents
        self.args = args
        self.iid_agents = args.iid_agents
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.order = [x for x in range(args.n_agents)]  if not args.turn_based and not args.random_order else None

        if self.args.ppo:
            self.learner = learner
            self.action_selector = action_REGISTRY[args.action_selector](args)

        else:
            self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        if self.order is None:
            avail_actions = avail_actions[:,ep_batch["order"][0, t_ep].to(int).squeeze()]
        else:
            avail_actions = avail_actions
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_actions(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        if self.order is None:
            order = ep_batch["order"][0, t].squeeze().to(int)
        else:
            order = self.order
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        if type(self.agent) is not list:

            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states[:,order])
            self.hidden_states = self.hidden_states.reshape(avail_actions.shape[0],avail_actions.shape[1], -1 )

        else:
            agent_inputs = agent_inputs.reshape(self.n_agents, int(agent_inputs.shape[0]/self.n_agents), -1)
            agent_outs = []
            states = []
            for i, a in enumerate(self.agent):
                outs = a(agent_inputs[i,:, :], self.hidden_states[i])
                agent_outs.append(outs[0])
                states.append(outs[1])

            self.hidden_states = states
            self.hidden_states = [x.reshape(avail_actions.shape[0], avail_actions.shape[1], -1) for x in self.hidden_states]

            agent_outs = th.stack(agent_outs)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions[:,order].reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        if type(self.agent) is not list:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

        else:  self.hidden_states = [x.init_hidden().unsqueeze(0).expand(batch_size, 1, -1) for x in self.agent]

    def parameters(self):
        return self.agent.parameters() if not self.iid_agents else [x.parameters() for x in self.agent]

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        agent_dict = th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage)
        self.agent.load_state_dict(agent_dict)

    def _build_agents(self, input_shape):
        if self.args.ppo:
            self.agent = self.learner.policy
        else:
            self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args) if not self.iid_agents else [agent_REGISTRY[self.args.agent](input_shape, self.args) for _ in range(self.args.n_agents)]

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        in_obs = batch["obs"][:, t]

        if self.order is None:
            in_obs = in_obs[:,batch["order"][0, t].squeeze().to(int)]
        else:
            in_obs = in_obs
        inputs.append(in_obs)  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
