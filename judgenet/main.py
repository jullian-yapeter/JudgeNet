import torch

from judgenet.config import CONFIG as cfg
from judgenet.modules.dataloader import get_split_dataloaders
from judgenet.stages.train import Trainer
from judgenet.stages.test import Tester


def experiment():
    # import data and instantiate train and val dataloaders
    data = torch.rand((1000, 51))
    train_loader, val_loader, test_loader = get_split_dataloaders(
        data, batch_size=cfg.batch_size, train=cfg.train_split, val=cfg.val_split)
    model = cfg.model_class(
        in_dim=cfg.in_dim,
        hidden_dim=cfg.hidden_dim,
        n_hidden_layers=cfg.n_hidden_layers,
        dropout_rate=cfg.dropout_rate)
    trained_model = Trainer(model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            epochs=cfg.epochs,
                            lr=cfg.lr).run()
    stats = Tester(model=trained_model,
                   test_loader=test_loader).run()
    print(stats)

if __name__=="__main__":
    print("starting")
    experiment()
    print("finished")