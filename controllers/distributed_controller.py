from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class DistributedMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        res = [self.agent[i](agent_inputs[i], self.hidden_states[i]) for i in range(self.args.n_agents)]
        agent_outs, self.hidden_states = [r[0] for r in res], [r[1] for r in res]
        agent_outs = th.stack(agent_outs, dim=1).reshape(ep_batch.batch_size * self.n_agents, -1)
          
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
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
        self.hidden_states = [self.agent[i].init_hidden().unsqueeze(0).expand(batch_size, -1, -1) for i in range(self.args.n_agents)] # bav

    def parameters(self):
        res = []
        for i in range(self.args.n_agents):
            res += list(self.agent[i].parameters())
        return res

    def load_state(self, other_mac):
        for i in range(self.args.n_agents):
            self.agent[i].load_state_dict(other_mac.agent[i].state_dict())

    def cuda(self):
        for i in range(self.args.n_agents):
            self.agent[i].cuda()

    def save_models(self, path):
        for i in range(self.args.n_agents):
            th.save(self.agent[i].state_dict(), "{}/agent_{}.th".format(path, i))

    def load_models(self, path):
        for i in range(self.args.n_agents):
            self.agent[i].load_state_dict(th.load("{}/agent_{}.th".format(path, i), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = [agent_REGISTRY[self.args.agent](input_shape, self.args) for _ in range(self.args.n_agents)]

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(1, device=batch.device).unsqueeze(0).expand(bs, self.args.n_agents, -1))
                 
        inputs = [th.cat([x[:, i].reshape(bs, -1) for x in inputs], dim=1) for i in range(self.args.n_agents)]
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += 1

        return input_shape
