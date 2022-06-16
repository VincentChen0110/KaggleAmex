import torch.nn as nn
import torch

class gru_model(nn.Module):
    def __init__(self, in_feats, hid_dim=256, activation=nn.ReLU()):
        super(gru_model, self).__init__()
        self.num_layers = 1
        self.hid_dim = hid_dim
        self.activation = activation
        self.hidden_state = None
        self.encode = nn.GRU(input_size=in_feats,
                             hidden_size=hid_dim,
                             num_layers=self.num_layers,
                             batch_first=True,
                             bidirectional=False)
        self.hidden = nn.Sequential(nn.Linear(hid_dim, 64),
                                    self.activation,
                                    nn.Linear(64, 32),
                                    self.activation)
        self.predict = nn.Linear(32, 2)
        
    def init_hidden(self, batch_size, device="cpu"):
        return torch.autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hid_dim)).to(device)
    
    def forward(self, x):
        _, h = self.encode(x, self.hidden_state)
        h = self.hidden(torch.squeeze(h))
        return self.predict(h)