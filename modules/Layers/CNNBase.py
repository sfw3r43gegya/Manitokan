import torch.nn as nn
import torch


"""CNN Modules and utils."""

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNLayer(nn.Module):
    def __init__(self, obs_shape, use_kern, hidden_size, use_orthogonal, device, activ='relu', kernel_size=3, stride=1):
        super(CNNLayer, self).__init__()

        active_func = {"tanh": nn.Tanh(),"relu": nn.ReLU(), "gelu":nn.GELU()}[activ]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(activ) if activ != 'gelu' else 1


        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]
        kern = hidden_size // 2 * (input_width - kernel_size + stride) * (input_height - kernel_size + stride) if use_kern else hidden_size // 2

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=hidden_size // 2,
                            kernel_size=kernel_size,
                            stride=stride,
                            device=device)
                  ),
            active_func,
            Flatten(),
            init_(nn.Linear(kern, #* (input_width - kernel_size + stride) * (input_height - kernel_size + stride), # changed from hidden_size // 2 * (input_width - kernel_size + stride) * (input_height - kernel_size + stride)
                            hidden_size,
                            device=device)
                  ),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size,
                            device=device)), active_func)

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        return x

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
class CNNBase(nn.Module):
    def __init__(self, args, obs_shape, device):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._activ = args.activation
        self.hidden_size = args.critic_hidden_dim
        self.device = device
        use_kern = args.env_args["completion_signal"] and args.env_args["key_signal"]

        self.cnn = CNNLayer(obs_shape, use_kern, self.hidden_size, self._use_orthogonal,activ= self._activ, device=self.device)

    def forward(self, x):
        x = self.cnn(x.to(torch.float32))
        return x