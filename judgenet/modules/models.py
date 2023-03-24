import torch
import torch.nn as nn
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from torchrl.modules import TanhNormal


class LinearNet(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 n_hidden_layers: int,
                 dropout_rate: float,
                 device=None):
        super().__init__()
        # self.device = torch.device(
        #     'mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = torch.device('cpu') if device is None else device
        net_layers = nn.ModuleList()
        net_layers.append(nn.Linear(in_dim, hidden_dim))
        net_layers.append(nn.LeakyReLU())
        net_layers.append(nn.Dropout(p=dropout_rate))
        net_layers.append(nn.LayerNorm(hidden_dim))
        for _ in range(n_hidden_layers):
            net_layers.append(nn.Linear(hidden_dim, hidden_dim))
            net_layers.append(nn.LeakyReLU())
            net_layers.append(nn.Dropout(p=dropout_rate))
            net_layers.append(nn.LayerNorm(hidden_dim))
        net_layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*net_layers).to(self.device)

        # Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
        self.apply(_weights_init)

    def forward(self, x):
        return self.net(x.to(self.device))

    def predict(self, x):
        with torch.no_grad():
            pred = torch.argmax(self.forward(x), dim=-1)
        return pred


class MLENet(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 n_hidden_layers: int,
                 dropout_rate: float,
                 device=None):
        super().__init__()
        self.device = torch.device('cpu') if device is None else device
        self.linearNet = LinearNet(in_dim,
                                   2,
                                   hidden_dim,
                                   n_hidden_layers,
                                   dropout_rate,
                                   device)

    def forward(self, x):
        x = self.linearNet(x)
        mu = x[:, 0].view(-1, 1)
        log_sigma = x[:, 1].view(-1, 1)
        return Independent(Normal(mu, log_sigma.exp()), 1)
    
    def loss(self, pred_dist, label):
        loss = -1 * pred_dist.log_prob(label).mean()
        return loss
    
    def predict(self, x):
        with torch.no_grad():
            x = self.linearNet(x)
            mu = x[:, 0].view(-1, 1)
            log_sigma = x[:, 1].view(-1, 1)
            dist = Independent(Normal(mu, log_sigma.exp()), 1)
        return mu, dist


class TanhMLENet(MLENet):
    
    def forward(self, x):
        x = self.linearNet(x)
        mu = x[:, 0].view(-1, 1)
        log_sigma = x[:, 1].view(-1, 1)
        return TanhNormal(mu, log_sigma.exp(), min=0.0, max=1.0)
