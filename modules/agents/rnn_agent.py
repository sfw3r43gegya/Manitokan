import torch.nn as nn
import torch.nn.functional as F




class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.device = args.device
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim, device=args.device)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim, device=args.device)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions, device=args.device)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.gelu(self.fc1(inputs.to(self.device)))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_in = h_in.to(self.device)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
