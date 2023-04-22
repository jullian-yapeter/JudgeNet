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
        super().__init__()
        self.device = torch.device('cpu') if device is None else device
        self.encoder = encoder.eval()
        self.predictor = predictor.eval()
        self.in_idxs = in_idxs

    def forward(self, x):
        x = x.to(self.device)
        return self.predictor(self.encoder(
            x[:, self.in_idxs[0]: self.in_idxs[1]]))

    def loss(self, outputs, labels):
        return self.predictor.loss(outputs, labels)

    @torch.no_grad()
    def predict(self, x):
        x = x.to(self.device)
        return self.predictor.predict(self.encoder(
            x[:, self.in_idxs[0]: self.in_idxs[1]]))


class KnowledgeDistiller(nn.Module):

    def __init__(
            self,
            student,
            teacher,
            student_in_idxs,
            teacher_in_idxs,
            temperature=7,
            alpha=0.3,
            device=None):
        super().__init__()
        self.device = torch.device('cpu') if device is None else device
        self.temperature = temperature
        self.alpha = alpha
        self.student = student
        self.teacher = teacher
        self.student_in_idxs = student_in_idxs
        self.teacher_in_idxs = teacher_in_idxs
        self.student.train()
        self.teacher.eval()

    def train(self, mode=True):
        self.training = mode
        if mode:
            self.student.train()
            self.teacher.eval()
        else:
            self.student.eval()
            self.teacher.eval()
        return self

    def forward(self, x):
        x = x.to(self.device)
        outputs = {}
        outputs["student"] = self.student(
            x[:, self.student_in_idxs[0]: self.student_in_idxs[1]])
        with torch.no_grad():
            outputs["teacher"] = self.teacher(
                x[:, self.teacher_in_idxs[0]: self.teacher_in_idxs[1]])
        return outputs

    def loss(self, outputs, label):
        p = F.log_softmax(outputs["student"]/self.temperature, dim=1)
        q = F.softmax(outputs["teacher"]/self.temperature, dim=1)
        l_kl = F.kl_div(p, q, size_average=False) * \
            (self.temperature**2) / outputs["student"].shape[0]
        l_ce = F.cross_entropy(outputs["student"], label)
        return l_kl * self.alpha + l_ce * (1. - self.alpha)


class Stage1(nn.Module):

    def __init__(
            self,
            mm_encoder: Encoder,
            mm_decoder: PredictorRegression,
            device=None):
        super().__init__()
        self.device = torch.device('cpu') if device is None else device
        self.mm_encoder = mm_encoder.train()
        self.mm_decoder = mm_decoder.train()

    def train(self, mode=True):
        self.training = mode
        if mode:
            self.mm_encoder.train()
            self.mm_decoder.train()
        else:
            self.mm_encoder.eval()
            self.mm_decoder.eval()
        return self

    def forward(self, x):
        outputs = {}
        x = x.to(self.device)
        outputs["x"] = x
        outputs["reconstruction"] = self.mm_decoder(self.mm_encoder(x))
        return outputs

    def loss(self, outputs, labels):
        return F.mse_loss(outputs["x"], outputs["reconstruction"])


class Stage2(nn.Module):

    def __init__(
            self,
            mm_encoder: Encoder,
            um_encoder: Encoder,
            um_in_idxs: Tuple,
            device=None):
        super().__init__()
        self.device = torch.device('cpu') if device is None else device
        self.mm_encoder = mm_encoder.eval()
        self.um_encoder = um_encoder.train()
        self.um_in_idxs = um_in_idxs

    def train(self, mode=True):
        self.training = mode
        if mode:
            self.mm_encoder.eval()
            self.um_encoder.train()
        else:
            self.mm_encoder.eval()
            self.um_encoder.eval()
        return self

    def forward(self, x):
        outputs = {}
        x = x.to(self.device)
        with torch.no_grad():
            outputs["mm_latent"] = self.mm_encoder(x)
        outputs["um_latent"] = self.um_encoder(
            x[:, self.um_in_idxs[0]: self.um_in_idxs[1]])
        return outputs

    def loss(self, outputs, labels):
        return F.mse_loss(outputs["mm_latent"].detach(), outputs["um_latent"])


class Stage3(nn.Module):

    def __init__(
            self,
            mm_encoder: Encoder,
            mm_predictor: Predictor,
            um_encoder: Encoder,
            um_in_idxs: Tuple,
            alpha: float = 0.1,
            device=None):
        super().__init__()
        self.device = torch.device('cpu') if device is None else device
        self.mm_encoder = mm_encoder.train()
        self.mm_predictor = mm_predictor.train()
        self.um_encoder = um_encoder.train()
        self.um_in_idxs = um_in_idxs
        self.alpha = alpha

    def train(self, mode=True):
        self.training = mode
        if mode:
            self.mm_encoder.train()
            self.mm_predictor.train()
            self.um_encoder.train()
        else:
            self.mm_encoder.eval()
            self.mm_predictor.eval()
            self.um_encoder.eval()
        return self

    def forward(self, x):
        x = x.to(self.device)
        outputs = {}
        mm_latent = self.mm_encoder(x)
        outputs["mm_latent"] = mm_latent
        outputs["mm_logits"] = self.mm_predictor(mm_latent)
        outputs["um_latent"] = self.um_encoder(
            x[:, self.um_in_idxs[0]: self.um_in_idxs[1]])
        return outputs

    def loss(self, outputs, labels):
        pred_loss = self.mm_predictor.loss(outputs["mm_logits"], labels)
        latent_loss = F.mse_loss(
            outputs["mm_latent"].detach(), outputs["um_latent"])
        return pred_loss + self.alpha * latent_loss


class Stage4(nn.Module):

    def __init__(
            self,
            mm_encoder: Encoder,
            um_encoder: Encoder,
            um_predictor: Predictor,
            um_in_idxs: Tuple,
            alpha: float = 0.1,
            device=None):
        super().__init__()
        self.device = torch.device('cpu') if device is None else device
        self.mm_encoder = mm_encoder.eval()
        self.um_encoder = um_encoder.train()
        self.um_predictor = um_predictor.train()
        self.um_in_idxs = um_in_idxs
        self.alpha = alpha

    def train(self, mode=True):
        self.training = mode
        if mode:
            self.mm_encoder.eval()
            self.um_encoder.train()
            self.um_predictor.train()
        else:
            self.mm_encoder.eval()
            self.um_encoder.eval()
            self.um_predictor.eval()
        return self

    def forward(self, x):
        x = x.to(self.device)
        outputs = {}
        with torch.no_grad():
            outputs["mm_latent"] = self.mm_encoder(x)
        um_latent = self.um_encoder(
            x[:, self.um_in_idxs[0]: self.um_in_idxs[1]])
        outputs["um_latent"] = um_latent
        outputs["um_logits"] = self.um_predictor(um_latent)
        return outputs

    def loss(self, outputs, labels):
        pred_loss = self.um_predictor.loss(outputs["um_logits"], labels)
        latent_loss = F.mse_loss(
            outputs["mm_latent"].detach(), outputs["um_latent"])
        return pred_loss + self.alpha * latent_loss
