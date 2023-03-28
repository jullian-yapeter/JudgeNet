import os

import torch

from judgenet.config_ted import CONFIG as cfg
from judgenet.modules.dataloader import get_split_dataloaders
from judgenet.stages.test import TesterTed
from judgenet.stages.train import Trainer
from judgenet.utils.general import Timer
from judgenet.values.constants import final_model_filename

def TED_experiment():
    train_loader, val_loader, test_loader = get_split_dataloaders(
        None, dataset_class=cfg.dataset_class, batch_size=cfg.batch_size, train=cfg.train_split, val=cfg.val_split)
    
    model = cfg.model_class(
        in_names=["lexical", "prosody"],
        in_dims=cfg.in_dims,
        out_dim=cfg.out_dim,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        n_hidden_layers=cfg.n_hidden_layers,
        dropout_rate=cfg.dropout_rate)
    
    Trainer(
        exp_name=cfg.exp_name,
        exp_dir=cfg.exp_dir,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr).run()
    trained_model = cfg.model_class(
        in_names=["lexical", "prosody"],
        in_dims=cfg.in_dims,
        out_dim=cfg.out_dim,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        n_hidden_layers=cfg.n_hidden_layers,
        dropout_rate=cfg.dropout_rate)
    trained_model.load_state_dict(
        torch.load(os.path.join(cfg.exp_dir, f"{cfg.exp_name}_{final_model_filename}")))
    # NOTE TODO, change test_loader back
    stats = TesterTed(
        exp_name=cfg.exp_name,
        exp_dir=cfg.exp_dir,
        model=trained_model,
        test_loader=train_loader).run()
    print(stats)

if __name__ == "__main__":
    with Timer(cfg.exp_name):
        TED_experiment()