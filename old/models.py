import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from torchrl.modules import TanhNormal


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


class JudgeNetEncoderDecoder(nn.Module):
    def __init__(
            self,
            in_dim: int,
            emb_dim: int,
            out_dim: int,
            hidden_dim: int,
            n_hidden_layers: int,
            dropout_rate: float,
            mode: str,
            device=None):
        super().__init__()
        self.device = torch.device('cpu') if device is None else device
        self.unimodal_indices = {}
        self.in_dim = in_dim
        self.encoder = LinearNet(
            in_dim=in_dim,
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
        self.mode = mode

    def forward(self, x):
        x = x.to(self.device)
        if self.mode == "lexical":
            x = x[:, :self.in_dim]
        return self.predictor(self.encoder(x))

    def loss(self, outputs, label):
        return self.ce_loss(
            outputs, label
        )

    def predict(self, x):
        with torch.no_grad():
            if self.mode == "lexical":
                x = x[:, :self.in_dim]
            emb = self.encoder(x)
            return torch.argmax(self.predictor(emb), dim=-1)


class MLENet(nn.Module):
    def __init__(
            self,
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


class JudgeNetAE(nn.Module):

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
        self.multimodal_encoder = LinearNet(
            in_dim=in_dim,
            out_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            dropout_rate=dropout_rate
        )
        self.multimodal_decoder = LinearNet(
            in_dim=emb_dim,
            out_dim=in_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            dropout_rate=dropout_rate
        )
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, x):
        x = x.to(self.device)
        return self.multimodal_decoder(self.multimodal_encoder(x))

    def loss(self, pred, original):
        return self.mse_loss(pred, original)
    

class JudgeNetDistill(nn.Module):

    def __init__(
            self,
            in_names: List[str],
            in_dims: List[int],
            emb_dim: int,
            hidden_dim: int,
            n_hidden_layers: int,
            dropout_rate: float,
            multimodal_encoder: nn.Module,
            device=None):
        super().__init__()
        self.device = torch.device('cpu') if device is None else device
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
        self.multimodal_encoder = multimodal_encoder.eval()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, x):
        x = x.to(self.device)
        outputs = {}
        for name in self.unimodal_encoders:
            start_idx, end_idx = self.unimodal_indices[name]
            unimodal_input = x[:, start_idx: end_idx]
            outputs[name] = self.unimodal_encoders[name](unimodal_input)
        with torch.no_grad():
            multimodal_emb = self.multimodal_encoder(x)
        outputs["multimodal"] = multimodal_emb
        return outputs

    def loss(self, outputs):
        loss = 0
        multimodal_emb = outputs["multimodal"].detach()
        for name in self.unimodal_encoders:
            loss += self.mse_loss(
                outputs[name], multimodal_emb
            )
        return loss


class KnowledgeDistiller(nn.Module):

    def __init__(self, student, teacher, device=None):
        super().__init__()
        self.device = torch.device('cpu') if device is None else device
        self.student = student
        self.teacher = teacher
        self.student.train()
        self.teacher.eval()

    def forward(self, x):
        x = x.to(self.device)
        outputs = {}
        outputs["student"] = self.student(x)
        with torch.no_grad():
            outputs["teacher"] = self.teacher(x)
        return outputs

    def loss(self, outputs, label, temperature=7, alpha=0.3):
        p = F.log_softmax(outputs["student"]/temperature, dim=1)
        q = F.softmax(outputs["teacher"]/temperature, dim=1)
        l_kl = F.kl_div(p, q, size_average=False) * \
            (temperature**2) / outputs["student"].shape[0]
        l_ce = F.cross_entropy(outputs["student"], label)
        return l_kl * alpha + l_ce * (1. - alpha)
    
    def predict(self, x):
        with torch.no_grad():
            return self.student.predict(x)


class JudgeNetSharedDecoder(nn.Module):

    def __init__(
            self,
            in_names: List[str],
            in_dims: List[int],
            emb_dim: int,
            out_dim: int,
            hidden_dim: int,
            n_hidden_layers: int,
            dropout_rate: float,
            multimodal_encoder: nn.Module,
            unimodal_encoders: nn.Module,
            device=None):
        super().__init__()
        self.device = torch.device('cpu') if device is None else device
        self.unimodal_indices = {}
        cur_index = 0
        for in_name, in_dim in zip(in_names, in_dims):
            self.unimodal_indices[in_name] = (cur_index, cur_index + in_dim)
        self.unimodal_encoders = unimodal_encoders.train()
        self.multimodal_encoder = multimodal_encoder.train()
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
        for name in self.unimodal_encoders:
            start_idx, end_idx = self.unimodal_indices[name]
            unimodal_input = x[:, start_idx: end_idx]
            outputs[name] = self.unimodal_encoders[name](unimodal_input)
        multimodal_emb = self.multimodal_encoder(x)
        outputs["multimodal"] = multimodal_emb
        outputs["prediction"] = self.predictor(multimodal_emb)
        return outputs

    def loss(self, outputs, label, alpha=0.1):
        prediction_loss = self.ce_loss(
            outputs["prediction"], label
        )
        teacher_student_loss = 0
        multimodal_emb = outputs["multimodal"].detach()
        for name in self.unimodal_encoders:
            teacher_student_loss += self.mse_loss(
                outputs[name], multimodal_emb
            )
        loss = prediction_loss + alpha * teacher_student_loss
        return loss
    
    def predict(self, x, mode="lexical"):
        with torch.no_grad():
            if mode == "multimodal":
                emb = self.multimodal_encoder(x)
            else:
                start_idx, end_idx = self.unimodal_indices[mode]
                x = x[:, start_idx: end_idx]
                emb = self.unimodal_encoders[mode](x)
            return torch.argmax(self.predictor(emb), dim=-1)


class JudgeNetFinetune(nn.Module):

    def __init__(
            self,
            in_names: List[str],
            in_dims: List[int],
            multimodal_encoder: nn.Module,
            unimodal_encoders: nn.Module,
            predictor: nn.Module,
            finetune_modality: str = "lexical",
            device=None):
        super().__init__()
        self.device = torch.device('cpu') if device is None else device
        self.finetune_modality = finetune_modality
        self.unimodal_indices = {}
        cur_index = 0
        for in_name, in_dim in zip(in_names, in_dims):
            self.unimodal_indices[in_name] = (cur_index, cur_index + in_dim)
        self.multimodal_encoder = multimodal_encoder.eval()
        self.unimodal_encoders = unimodal_encoders.train()
        self.predictor = predictor.train()
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, x):
        x = x.to(self.device)
        outputs = {}
        with torch.no_grad():
            multimodal_emb = self.multimodal_encoder(x)
        start_idx, end_idx = self.unimodal_indices[self.finetune_modality]
        unimodal_input = x[:, start_idx: end_idx]
        unimodal_emb = self.unimodal_encoders[self.finetune_modality](
            unimodal_input)
        outputs["multimodal"] = multimodal_emb
        outputs["unimodal"] = unimodal_emb
        outputs["prediction"] = self.predictor(unimodal_emb)
        return outputs

    def loss(self, outputs, label, alpha=0.2):
        prediction_loss = self.ce_loss(
            outputs["prediction"], label
        )
        teacher_student_loss = self.mse_loss(
            outputs["unimodal"], outputs["multimodal"].detach()
        )
        loss = prediction_loss + alpha * teacher_student_loss
        return loss

    def predict(self, x, mode="lexical"):
        with torch.no_grad():
            if mode == "multimodal":
                emb = self.multimodal_encoder(x)
            else:
                start_idx, end_idx = self.unimodal_indices[mode]
                x = x[:, start_idx: end_idx]
                emb = self.unimodal_encoders[mode](x)
            return torch.argmax(self.predictor(emb), dim=-1)