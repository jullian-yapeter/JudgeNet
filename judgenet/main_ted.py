from judgenet.config_ted import CONFIG as cfg
from judgenet.modules.dataloader import get_split_dataloaders
from judgenet.stages.pretrain import MultimodalPretrainer, UnimodalPretrainer
from judgenet.stages.test import TesterClassification
from judgenet.stages.train import Trainer
from judgenet.utils.general import Timer

from judgenet.modules.models import LinearNet
from torch import nn

def TED_experiment():

    # Initialize dataloaders
    train_loader, val_loader, test_loader = get_split_dataloaders(
        None,
        dataset_class=cfg.dataset_class,
        batch_size=cfg.batch_size,
        train=cfg.train_split,
        val=cfg.val_split
    )

    if cfg.unimodal_baseline:
        print("Running unimodal baseline")
        baseline_net = cfg.baseline_class(
            in_dim=cfg.lexical_dim,
            emb_dim=cfg.emb_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.out_dim,
            n_hidden_layers=cfg.n_hidden_layers,
            dropout_rate=cfg.dropout_rate,
            mode="lexical"
        )
        baseline_net = Trainer(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            model=baseline_net,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg.epochs,
            lr=cfg.lr).run()
        stats = TesterClassification(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            model=baseline_net.eval(),
            test_loader=test_loader).run()
        print(stats)
    elif cfg.multimodal_baseline:
        print("Running multimodal baseline")
        baseline_net = cfg.baseline_class(
            in_dim=sum(cfg.in_dims),
            emb_dim=cfg.emb_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.out_dim,
            n_hidden_layers=cfg.n_hidden_layers,
            dropout_rate=cfg.dropout_rate,
            mode="multimodal"
        )
        baseline_net = Trainer(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            model=baseline_net,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg.epochs,
            lr=cfg.lr).run()
        stats = TesterClassification(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            model=baseline_net.eval(),
            test_loader=test_loader).run()
        print(stats)
    else:

        if cfg.use_pretrain:
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
        else:
            print("Skipping pretraining")
            # Testing out removing the pretraining
            multimodal_encoder = LinearNet(
                in_dim=sum(cfg.in_dims),
                out_dim=cfg.emb_dim,
                hidden_dim=cfg.hidden_dim,
                n_hidden_layers=cfg.n_hidden_layers,
                dropout_rate=cfg.dropout_rate,
            )

            unimodal_encoders = nn.ModuleDict()
            for in_name, in_dim in zip(cfg.in_names, cfg.in_dims):
                unimodal_encoders[in_name] = LinearNet(
                    in_dim=in_dim,
                    out_dim=cfg.emb_dim,
                    hidden_dim=cfg.hidden_dim,
                    n_hidden_layers=cfg.n_hidden_layers,
                    dropout_rate=cfg.dropout_rate
                )

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
        stats = TesterClassification(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            model=predictor_net.eval(),
            test_loader=test_loader).run()
        print(stats)

        if cfg.use_finetune:
            # Finetune unimodal networks
            multimodal_encoder = predictor_net.multimodal_encoder
            unimodal_encoders = predictor_net.unimodal_encoders
            predictor = predictor_net.predictor
            
            finetune_net = cfg.finetune_class(
                in_names=cfg.in_names,
                in_dims=cfg.in_dims,
                multimodal_encoder=multimodal_encoder,
                unimodal_encoders=unimodal_encoders,
                predictor=predictor,
                finetune_modality=cfg.finetune_modality
            )
            finetune_net = Trainer(
                exp_name=cfg.exp_name,
                exp_dir=cfg.exp_dir,
                model=finetune_net,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=cfg.epochs,
                lr=cfg.lr).run()
            stats = TesterClassification(
                exp_name=cfg.exp_name,
                exp_dir=cfg.exp_dir,
                model=finetune_net.eval(),
                test_loader=test_loader).run()
        else:
            print("Skipping fine tuning")
        print(stats)


if __name__ == "__main__":
    with Timer(cfg.exp_name):
        TED_experiment()