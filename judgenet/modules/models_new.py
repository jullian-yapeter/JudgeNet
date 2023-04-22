from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNet(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_dim: int,
            n_hidden_layers: int,
            dropout_rate: float,
            device=None):
        super().__init__()
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
                nn.init.kaiming_uniform_(m.weight)
                m.bias.data.zero_()
        self.apply(_weights_init)

    def forward(self, x):
        return self.net(x.to(self.device))
    

class Encoder(nn.Module):

    def __init__(
            self,
            in_dim: int,
            emb_dim: int,
            hidden_dim: int,
            n_hidden_layers: int,
            dropout_rate: float,
            device=None):
        super().__init__()
        self.device = torch.device('cpu') if device is None else device
        self.linear_net = LinearNet(
            in_dim=in_dim,
            out_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            dropout_rate=dropout_rate
        )

    def forward(self, x):
        return self.linear_net(x.to(self.device))
    

class Predictor(nn.Module):

    def __init__(
            self,
            emb_dim: int,
            out_dim: int,
            hidden_dim: int,
            n_hidden_layers: int,
            dropout_rate: float,
            device=None):
        super().__init__()
        self.device = torch.device('cpu') if device is None else device
        self.linear_net = LinearNet(
            in_dim=emb_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            dropout_rate=dropout_rate
        )

    def forward(self, x):
        return self.linear_net(x.to(self.device))
    

class PredictorClassification(Predictor):

    def loss(self, outputs, labels):
        return F.cross_entropy(outputs, labels)

    @torch.no_grad()
    def predict(self, x):
        return torch.argmax(self.linear_net(x), dim=-1)
        

class PredictorRegression(Predictor):

    def loss(self, outputs, labels):
        return F.mse_loss(outputs, labels)

    @torch.no_grad()
    def predict(self, x):
        return self.linear_net(x)
    

class EncoderPredictor(nn.Module):

    def __init__(
            self,
            encoder: Encoder,
            predictor: Predictor,
            in_idxs: Tuple,
            device=None):
        super.__init__()
        self.device = torch.device('cpu') if device is None else device
        self.encoder = encoder.eval()
        self.predictor = predictor.eval()
        self.in_idxs = in_idxs

    def forward(self, x):
        x = x.to(self.device)
        return self.predictor(self.encoder(
            x[:, self.in_idxs[0]: self.in_idxs[1]]))

    @torch.no_grad()
    def predict(self, x):
        x = x.to(self.device)
        return self.predictor.predict(self.encoder(
            x[:, self.in_idxs[0]: self.in_idxs[1]]))


class Stage1(nn.Module):

    def __init__(
            self,
            mm_encoder: Encoder,
            mm_predictor: PredictorRegression,
            device=None):
        super.__init__()
        self.device = torch.device('cpu') if device is None else device
        self.mm_encoder = mm_encoder.train()
        self.mm_predictor = mm_predictor.train()

    def forward(self, x):
        outputs = {}
        x = x.to(self.device)
        outputs["x"] = x
        outputs["reconstruction"] = self.mm_predictor(self.mm_encoder(x))
        return outputs

    def loss(self, outputs, labels):
        return F.mse_loss(outputs["x"], outputs["reconstruction"])


class Stage2(nn.Module):

    def __init__(
            self,
            mm_encoder: Encoder,
            um_encoder: Encoder,
            um_idxs: Tuple,
            device=None):
        super.__init__()
        self.device = torch.device('cpu') if device is None else device
        self.mm_encoder = mm_encoder.eval()
        self.um_encoder = um_encoder.train()
        self.um_idxs = um_idxs

    def forward(self, x):
        outputs = {}
        x = x.to(self.device)
        with torch.no_grad():
            outputs["mm_latent"] = self.mm_encoder(x)
        outputs["um_latent"] = self.um_encoder(
            x[:, self.um_idxs[0], self.um_idxs[1]])
        return outputs

    def loss(self, outputs, labels):
        return F.mse_loss(outputs["mm_latent"].detach(), outputs["um_latent"])


class Stage3(nn.Module):

    def __init__(
            self,
            mm_encoder: Encoder,
            mm_predictor: Predictor,
            um_encoder: Encoder,
            um_idxs: Tuple,
            alpha: float = 0.1,
            device=None):
        super.__init__()
        self.device = torch.device('cpu') if device is None else device
        self.mm_encoder = mm_encoder.train()
        self.mm_predictor = mm_predictor.train()
        self.um_encoder = um_encoder.train()
        self.um_idxs = um_idxs
        self.alpha = alpha

    def forward(self, x):
        x = x.to(self.device)
        outputs = {}
        mm_latent = self.mm_encoder(x)
        outputs["mm_latent"] = mm_latent
        outputs["mm_logits"] = self.mm_predictor(mm_latent)
        outputs["um_latent"] = self.um_encoder(
            x[:, self.um_idxs[0], self.um_idxs[1]])
        return outputs

    def loss(self, outputs, labels):
        pred_loss = self.mm_predictor.loss(outputs["mm_logits"], labels)
        latent_loss = F.mse_loss(outputs["mm_latent"].detach(), outputs["um_latent"])
        return pred_loss + self.alpha * latent_loss


class Stage4(nn.Module):

    def __init__(
            self,
            mm_encoder: Encoder,
            um_encoder: Encoder,
            um_predictor: Predictor,
            um_idxs: Tuple,
            alpha: float = 0.1,
            device=None):
        super.__init__()
        self.device = torch.device('cpu') if device is None else device
        self.mm_encoder = mm_encoder.eval()
        self.um_encoder = um_encoder.train()
        self.um_predictor = um_predictor.train()
        self.um_idxs = um_idxs
        self.alpha = alpha

    def forward(self, x):
        x = x.to(self.device)
        outputs = {}
        with torch.no_grad():
            outputs["mm_latent"] = self.mm_encoder(x)
        um_latent = self.um_encoder(
            x[:, self.um_idxs[0], self.um_idxs[1]])
        outputs["um_latent"] = um_latent
        outputs["um_logits"] = self.um_predictor(um_latent)
        return outputs

    def loss(self, outputs, labels):
        pred_loss = self.um_predictor.loss(outputs["um_logits"], labels)
        latent_loss = F.mse_loss(outputs["mm_latent"].detach(), outputs["um_latent"])
        return pred_loss + self.alpha * latent_loss
