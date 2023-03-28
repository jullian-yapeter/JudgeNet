from typing import List

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


class JudgeNetSharedDecoder(nn.Module):

    def __init__(self,
                 in_names: List[str],
                 in_dims: List[int],
                 out_dim: int,
                 emb_dim: int,
                 hidden_dim: int,
                 n_hidden_layers: int,
                 dropout_rate: float,
                 device=None):
        assert len(in_names) == len(in_dims), "Length of names does not match length of in dims"
        super().__init__()
        self.device = torch.device('cpu') if device is None else device
        self.in_names = in_names
        self.unimodal_encoders = nn.ModuleDict()
        self.unimodal_indices = {}
        cur_index = 0
        for in_name, in_dim in zip(in_names, in_dims):
            self.unimodal_indices[in_name] = (cur_index, cur_index + in_dim)
            cur_index += in_dim
            self.unimodal_encoders[in_name] = LinearNet(
                in_dim=in_dim,
                out_dim=emb_dim,
                hidden_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                dropout_rate=dropout_rate
            )
        self.multimodal_encoder = LinearNet(
            in_dim=sum(in_dims),
            out_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            dropout_rate=dropout_rate
        )
        self.predictor = LinearNet(
            in_dim=emb_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            dropout_rate=dropout_rate
        )
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, x):
        x = x.to(self.device)
        outputs = {}
        for name in self.in_names:
            start_idx, end_idx = self.unimodal_indices[name]
            unimodal_input = x[:, start_idx: end_idx]
            outputs[name] = self.unimodal_encoders[name](unimodal_input)
        multimodal_emb = self.multimodal_encoder(x)
        outputs["multimodal"] = multimodal_emb
        outputs["prediction"] = self.predictor(multimodal_emb)
        return outputs

    def loss(self, outputs, label, alpha=0.5):
        prediction_loss = self.ce_loss(
            outputs["prediction"].to(self.device), label.to(self.device)
        )
        teacher_student_loss = 0
        multimodal_emb = outputs["multimodal"].detach()
        for name in self.in_names:
            teacher_student_loss += self.mse_loss(
                outputs[name].to(self.device), multimodal_emb
            )
        loss = alpha * prediction_loss + (1 - alpha) * teacher_student_loss
        return loss
