import torch
import torch.nn as nn
import torch as th
import numpy as np
from modules.Layers.CNNBase import CNNBase
from modules.Layers.MLPBase import MLPBase
from modules.Layers.RNNLayer import RNNLayer
from modules.Layers.act import ACTLayer as act


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=th.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.rnn_hidden_dim
        self.device = device
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N

        self.tpdv = dict(dtype=float, device=device)
        self.use_cnn = args.use_cnn
        self.task_sig = args.env_args["completion_signal"]

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if args.use_cnn else MLPBase

        if args.use_cnn and len(obs_shape) == 1:
            if self.task_sig :
                mid_shape = int(obs_shape[0]/3)
                mid_shape_1 = int((mid_shape/2))
                obs_shape = th.zeros(1,
                                 mid_shape ,
                                 3).shape

            else:
                mid_shape = int(obs_shape[0]/3)
                mid_shape_1 = int((mid_shape/3))
                obs_shape = th.zeros(int(mid_shape/mid_shape_1),
                                 mid_shape_1 ,
                                 3).shape

        self.base = base(args, obs_shape, device)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal, arch = args.policy_arch)


        self.act = act(action_space, self.hidden_size, self._use_orthogonal, self._gain, args.la_update)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if len(obs.shape) == 2 and self.use_cnn:

            if self.task_sig:
                mid_shape = int(obs.shape[1] / 3)
                mid_shape_1 = int((mid_shape / 2))
                obs = obs.reshape(obs.shape[0], 1, mid_shape, 3)

            else:
                if obs.shape[1] == 27 or obs.shape[1] == 54:
                    mid_shape = int(obs.shape[1] / 3)
                    mid_shape_1 = int((mid_shape / 3))
                    obs = obs.reshape(obs.shape[0], -1, mid_shape_1, 3)
                else:
                    mid_shape = int(obs.shape[1] / 3)
                    mid_shape_1 = int((mid_shape / 3))
                    obs = obs.reshape(obs.shape[0], int(mid_shape), mid_shape_1, 3)


        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            if len(actor_features.shape) == 1:
                actor_features = actor_features.unsqueeze(0)

            if len(masks.shape) == 1:
                masks = masks.unsqueeze(0)

            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks.swapaxes(0,1))

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def select_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        if len(obs.shape) == 2 and self.use_cnn:
            if self.task_sig:
                mid_shape = int(obs.shape[1] / 3)
                mid_shape_1 = int((mid_shape / 2))
                obs = obs.reshape(obs.shape[0], 1, mid_shape, 3)

            else:
                mid_shape = int(obs.shape[1] / 3)
                mid_shape_1 = int((mid_shape / 3))
                obs = obs.reshape(obs.shape[0], -1, mid_shape_1, 3)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            if len(actor_features.shape) == 1:
                actor_features = actor_features.unsqueeze(0)

            if len(masks.shape) == 1:
                masks = masks.unsqueeze(0)

            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        max_acts = torch.argmax(action,dim=1).unsqueeze(1)
        action_log_probs, dist_entropy = self.act.select_action(actor_features,
                                                                   max_acts,
                                                                  available_actions= available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy

    def get_entropy(self, obs, aval, rnn_states, masks):

        if len(obs.shape) == 2 and self.use_cnn:

            if self.task_sig:
                mid_shape = int(obs.shape[1] / 3)
                mid_shape_1 = int((mid_shape / 2))
                obs = obs.reshape(obs.shape[0], -1, mid_shape_1, 3)

            else:
                if obs.shape[1] == 27:
                    mid_shape = int(obs.shape[1] / 3)
                    mid_shape_1 = int((mid_shape / 3))
                    obs = obs.reshape(obs.shape[0], -1, mid_shape_1, 3)
                else:
                    mid_shape = int(obs.shape[1] / 3)
                    mid_shape_1 = int((mid_shape / 3))
                    obs = obs.reshape(obs.shape[0], int(mid_shape), mid_shape_1, 3)




        actor_features = self.base(obs.double())

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            if len(actor_features.shape) == 1:
                actor_features = actor_features.unsqueeze(0)

            if len(masks.shape) == 1:
                masks = masks.unsqueeze(0)

            actor_features, rnn_states = self.rnn(x=actor_features,hxs= torch.from_numpy(rnn_states), masks= torch.from_numpy(masks.swapaxes(0,1)))

        entropy = self.act.get_entropy(x=actor_features, available_actions=aval)

        return entropy


def check(input):
    output = th.from_numpy(input) if type(input) == np.ndarray else input

    return output

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        obs_shape = [obs_space]
    return obs_shape