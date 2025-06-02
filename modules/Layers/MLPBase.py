import torch.nn
import torch.nn as nn
import copy

"""MLP modules."""

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, activation, device):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N
        self.device = device
        acts = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'gelu': nn.GELU(),}
        active_func = acts[activation]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        if activation == 'gelu':

            gain = nn.init.calculate_gain('relu')
        else:
            gain = nn.init.calculate_gain(activation)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size)).to(self.device, dtype=float)
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size)).to(self.device, dtype=float)
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, device, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()
        self.device = device
        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal

        self.activation = args.activation
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.critic_hidden_dim

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim).to(self.device, dtype=float)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self.activation, self.device)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x