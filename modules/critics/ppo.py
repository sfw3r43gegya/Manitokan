import torch as th
import torch.nn as nn
import numpy as np
from modules.Layers.CNNBase import CNNBase
from modules.Layers.MLPBase import MLPBase
from modules.Layers.RNNLayer import RNNLayer
from modules.Layers.popart import PopArt

class PPOCritic(nn.Module):
    def __init__(self,  args, cent_obs_space, device=th.device("cpu")):
        super(PPOCritic, self).__init__()
        self.args = args
        self.device = device
        self.hidden_size = args.critic_hidden_dim
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.arch = args.policy_arch
        self._use_orthogonal = args.use_orthogonal
       # input_shape = self._get_input_shape(scheme)
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
      #  self.input_shape = input_shape
        self.output_type = "q"
        self.n_linear_layers = args.layer_N - 1
        self.activations =  {'relu': th.nn.ReLU(), 'gelu': th.nn.GELU(), 'tanh': th.nn.Tanh()}
        self.tpdv = dict(dtype=float, device=device)
        self.use_cnn = args.use_cnn
        self.share_buff = args.share_buffer
        self.task_sig = args.env_args["completion_signal"]

        # Set up network layers
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)

        base = CNNBase if args.use_cnn else MLPBase

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]


        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        if args.use_cnn and len(cent_obs_shape) == 1 and args.share_buffer:

            if self.task_sig :
                mid_shape = int(cent_obs_shape[0]/3)
                mid_shape_1 = int((mid_shape/2))
                cent_obs_shape= th.zeros(int(mid_shape/mid_shape_1),
                                 mid_shape_1 ,
                                 3).shape

            else:
                mid_shape = int(cent_obs_shape[0]/3)
                mid_shape_1 = int((mid_shape/3))
                cent_obs_shape = th.zeros(self.n_agents*int(mid_shape/mid_shape_1),
                                 mid_shape_1 ,
                                 3).shape

        elif args.use_cnn and len(cent_obs_shape) == 1:
            if self.task_sig :
                mid_shape = int(cent_obs_shape[0]/3)
                mid_shape_1 = int((mid_shape/2))
                cent_obs_shape= th.zeros(int(mid_shape/mid_shape_1),
                                 mid_shape_1 ,
                                 3).shape

            else:
                mid_shape = int(cent_obs_shape[0]/3)
                mid_shape_1 = int((mid_shape/3))
                cent_obs_shape = th.zeros(mid_shape_1 ,
                                          int(mid_shape/mid_shape_1),
                                 3).shape

        self.base = base(args =self.args,
                         obs_shape=cent_obs_shape, device=self.device)



        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size,
                                self.hidden_size,
                                self._recurrent_N,
                                self._use_orthogonal,
                                self.arch)


        self.fc3 = nn.Linear(args.critic_hidden_dim, 1)# HACK!

        self.to(device)

    def forward(self, cent_obs, rnn_states = None, masks = None, t=None):

        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if len(cent_obs.shape) == 2 and self.use_cnn:
            if self.task_sig:
                mid_shape = int(cent_obs.shape[1]/3)

                mid_shape_1 = int((mid_shape/2))
                cent_obs = cent_obs.reshape(int(cent_obs.shape[0]), -1, mid_shape_1, 3)

            else:
                mid_shape = int(cent_obs.shape[0]/3)
                mid_shape_1 = int((mid_shape/3))
                cent_obs = cent_obs.reshape(int(cent_obs.shape[0]),
                                            -1,
                                 3 ,
                                 3)



        inputs = self.base(cent_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy: # consider recurrence
            if len(inputs.shape) == 1:
                inputs = inputs.unsqueeze(0)

            if len(masks.shape) == 1:
                masks = masks.unsqueeze(0)

            critic_features, rnn_states = self.rnn(inputs, rnn_states, masks)
            q = self.fc3(critic_features)

            return q, rnn_states

        else:
            q = self.fc3(inputs)

            return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observation
        inputs.append(batch["obs"][:, ts])

        # actions (masked out by agent)
        actions = batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

        # last actions
        if t == 0:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))

        elif isinstance(t, int):
            inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))

        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        input_shape += scheme["obs"]["vshape"]
        # actions and last actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents * 2
        # agent id
        input_shape += self.n_agents
        return input_shape


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        obs_shape = [obs_space]
    return obs_shape


def check(input):
    if type(input) == np.ndarray:
        return th.from_numpy(input)

    else:
        return input


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module