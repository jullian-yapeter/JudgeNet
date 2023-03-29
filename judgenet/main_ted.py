import os

import torch

from judgenet.config_ted import CONFIG as cfg
from judgenet.modules.dataloader import get_split_dataloaders
from judgenet.stages.test import TesterTed
from judgenet.stages.train import Trainer
from judgenet.stages.pretrain import MultimodalPretrainer, UnimodalPretrainer
from judgenet.utils.general import Timer

def TED_experiment():

    # Initialize dataloaders
    train_loader, val_loader, test_loader = get_split_dataloaders(
        None,
        dataset_class=cfg.dataset_class,
        batch_size=cfg.batch_size,
        train=cfg.train_split,
        val=cfg.val_split
    )
    
    # Initialize and pre-train multimodal components
    multimodal_ae = cfg.multimodal_ae_class(
        in_dim=sum(cfg.in_dims),
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        n_hidden_layers=cfg.n_hidden_layers,
        dropout_rate=cfg.dropout_rate,
    )
    multimodal_ae = MultimodalPretrainer(
        exp_name=cfg.exp_name,
        exp_dir=cfg.exp_dir,
        model=multimodal_ae,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr
    ).run()

    # Initialize and pre-train unimodal components
    multimodal_encoder = multimodal_ae.multimodal_encoder.eval()

    distill_net = cfg.distill_net_class(
        in_names=cfg.in_names,
        in_dims=cfg.in_dims,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        n_hidden_layers=cfg.n_hidden_layers,
        dropout_rate=cfg.dropout_rate,
        multimodal_encoder=multimodal_encoder
    )
    distill_net = UnimodalPretrainer(
        exp_name=cfg.exp_name,
        exp_dir=cfg.exp_dir,
        model=distill_net,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr
    ).run()

    # Train the multimodal model on the downstream task
    multimodal_encoder = distill_net.multimodal_encoder.train()
    unimodal_encoders = distill_net.unimodal_encoders.train()

    predictor_net = cfg.predictor_class(
        in_names=cfg.in_names,
        in_dims=cfg.in_dims,
        out_dim=cfg.out_dim,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        n_hidden_layers=cfg.n_hidden_layers,
        dropout_rate=cfg.dropout_rate,
        multimodal_encoder=multimodal_encoder,
        unimodal_encoders=unimodal_encoders,
    )
    predictor_net = Trainer(
        exp_name=cfg.exp_name,
        exp_dir=cfg.exp_dir,
        model=predictor_net,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr).run()
    stats = TesterTed(
        exp_name=cfg.exp_name,
        exp_dir=cfg.exp_dir,
        model=predictor_net.eval(),
        test_loader=test_loader).run()
    print(stats)

    # Finetune unimodal networks
    unimodal_encoders = predictor_net.unimodal_encoders
    

if __name__ == "__main__":
    with Timer(cfg.exp_name):
        TED_experiment()