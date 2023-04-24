import copy
import os

from torch.utils.data import DataLoader

from judgenet.modules.dataloader import get_split_dataloaders
from judgenet.modules.models import *
from judgenet.utils.results import Results
from judgenet.values.constants import test_metrics_filename


from config_mit_lexical import CONFIG as cfg


# Exp Results
exp_results = Results()

for _ in range(cfg.n_runs):
    # Instantiate Dataloader
    train_loader, val_loader, test_loader = get_split_dataloaders(
        None,
        dataset_class=cfg.dataset_class,
        batch_size=cfg.batch_size,
        train=cfg.train_split,
        val=cfg.val_split
    )
    if hasattr(cfg, "pretrain_dataset_class"):
        pretrain_loader = DataLoader(
            cfg.pretrain_dataset_class(), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    else:
        pretrain_loader = train_loader

    if cfg.run_baselines:

        # Run Unimodal Baseline
        um_encoder = Encoder(
            in_dim=cfg.um_in_dim,
            emb_dim=cfg.emb_dim,
            hidden_dim=cfg.hidden_dim,
            n_hidden_layers=cfg.n_hidden_layers,
            dropout_rate=cfg.dropout_rate
        )
        predictor = cfg.predictor_class(
            emb_dim=cfg.emb_dim,
            out_dim=cfg.out_dim,
            hidden_dim=cfg.hidden_dim,
            n_hidden_layers=cfg.n_hidden_layers,
            dropout_rate=cfg.dropout_rate
        )
        um_model = EncoderPredictor(
            encoder=um_encoder,
            predictor=predictor,
            in_idxs=cfg.um_in_idxs
        )
        um_model = cfg.trainer_class(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            stage_name="um_baseline",
            model=um_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg.epochs,
            lr=cfg.lr).run()
        stats = cfg.tester_class(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            model=um_model,
            test_loader=test_loader
        ).run()
        print(f"um_baseline: \n{stats}")
        exp_results.update_results("um_baseline", stats)

        # Run Multimodal Baseline
        mm_encoder = Encoder(
            in_dim=cfg.mm_in_dim,
            emb_dim=cfg.emb_dim,
            hidden_dim=cfg.hidden_dim,
            n_hidden_layers=cfg.n_hidden_layers,
            dropout_rate=cfg.dropout_rate
        )
        predictor = cfg.predictor_class(
            emb_dim=cfg.emb_dim,
            out_dim=cfg.out_dim,
            hidden_dim=cfg.hidden_dim,
            n_hidden_layers=cfg.n_hidden_layers,
            dropout_rate=cfg.dropout_rate
        )
        mm_model = EncoderPredictor(
            encoder=mm_encoder,
            predictor=predictor,
            in_idxs=cfg.mm_in_idxs
        )
        mm_model = cfg.trainer_class(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            stage_name="mm_baseline",
            model=mm_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg.epochs,
            lr=cfg.lr).run()
        stats = cfg.tester_class(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            model=mm_model,
            test_loader=test_loader
        ).run()
        print(f"mm_baseline: \n{stats}")
        exp_results.update_results("mm_baseline", stats)

        # Run Knowledge Distillation Baseline
        um_encoder = Encoder(
            in_dim=cfg.um_in_dim,
            emb_dim=cfg.emb_dim,
            hidden_dim=cfg.hidden_dim,
            n_hidden_layers=cfg.n_hidden_layers,
            dropout_rate=cfg.dropout_rate
        )
        predictor = cfg.predictor_class(
            emb_dim=cfg.emb_dim,
            out_dim=cfg.out_dim,
            hidden_dim=cfg.hidden_dim,
            n_hidden_layers=cfg.n_hidden_layers,
            dropout_rate=cfg.dropout_rate
        )
        um_model = EncoderPredictor(
            encoder=um_encoder,
            predictor=predictor,
            in_idxs=cfg.um_in_idxs
        )
        kd_model = cfg.kd_class(
            student=um_model,
            teacher=mm_model,
            temperature=cfg.kd_temperature,
            alpha=cfg.kd_alpha
        )
        kd_model = cfg.trainer_class(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            stage_name="kd_baseline",
            model=kd_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg.epochs,
            lr=cfg.lr).run()
        stats = cfg.tester_class(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            model=kd_model.student,
            test_loader=test_loader
        ).run()
        print(f"kd_baseline: \n{stats}")
        exp_results.update_results("kd_baseline", stats)

    # Instantiate Encoders and Predictors
    mm_encoder = Encoder(
        in_dim=cfg.mm_in_dim,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        n_hidden_layers=cfg.n_hidden_layers,
        dropout_rate=cfg.dropout_rate
    )

    mm_decoder = PredictorRegression(
        emb_dim=cfg.emb_dim,
        out_dim=cfg.mm_in_dim,
        hidden_dim=cfg.hidden_dim,
        n_hidden_layers=cfg.n_hidden_layers,
        dropout_rate=cfg.dropout_rate
    )

    um_encoder = Encoder(
        in_dim=cfg.um_in_dim,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        n_hidden_layers=cfg.n_hidden_layers,
        dropout_rate=cfg.dropout_rate
    )

    predictor = cfg.predictor_class(
        emb_dim=cfg.emb_dim,
        out_dim=cfg.out_dim,
        hidden_dim=cfg.hidden_dim,
        n_hidden_layers=cfg.n_hidden_layers,
        dropout_rate=cfg.dropout_rate
    )

    if cfg.use_pretrain:

        # Stage 1
        stage1 = Stage1(
            mm_encoder=mm_encoder,
            mm_decoder=mm_decoder,
        )
        stage1 = cfg.trainer_class(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            stage_name="stage1",
            model=stage1,
            train_loader=pretrain_loader,
            val_loader=val_loader,
            epochs=cfg.epochs,
            lr=cfg.lr).run()
        mm_encoder = stage1.mm_encoder
        mm_decoder = stage1.mm_decoder

        # Stage 2
        stage2 = Stage2(
            mm_encoder=mm_encoder,
            um_encoder=um_encoder,
            um_in_idxs=cfg.um_in_idxs,
        )
        stage2 = cfg.trainer_class(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            stage_name="stage2",
            model=stage2,
            train_loader=pretrain_loader,
            val_loader=val_loader,
            epochs=cfg.epochs,
            lr=cfg.lr).run()
        mm_encoder = stage2.mm_encoder
        um_encoder = stage2.um_encoder

    # Stage 3
    stage3 = Stage3(
        mm_encoder=mm_encoder,
        mm_predictor=predictor,
        um_encoder=um_encoder,
        um_in_idxs=cfg.um_in_idxs,
        alpha=cfg.stage3_alpha
    )
    stage3 = cfg.trainer_class(
        exp_name=cfg.exp_name,
        exp_dir=cfg.exp_dir,
        stage_name="stage3",
        model=stage3,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr).run()
    mm_encoder = stage3.mm_encoder
    predictor = stage3.mm_predictor
    um_encoder = stage3.um_encoder

    # Post Stage 3 Test
    stage_3_model = EncoderPredictor(
        encoder=copy.deepcopy(um_encoder),
        predictor=copy.deepcopy(predictor),
        in_idxs=cfg.um_in_idxs
    )
    stats = cfg.tester_class(
        exp_name=cfg.exp_name,
        exp_dir=cfg.exp_dir,
        model=stage_3_model,
        test_loader=test_loader
    ).run()
    print(f"post-stage3: \n{stats}")
    exp_results.update_results("post-stage3", stats)

    if cfg.use_finetune:
        # Stage 4 No Regularization
        stage4_NoReg = Stage4(
            mm_encoder=copy.deepcopy(mm_encoder),
            um_encoder=copy.deepcopy(um_encoder),
            um_predictor=copy.deepcopy(predictor),
            um_in_idxs=cfg.um_in_idxs,
            alpha=0.0
        )
        stage4_NoReg = cfg.trainer_class(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            stage_name="stage4-NoReg",
            model=stage4_NoReg,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg.epochs,
            lr=cfg.lr).run()

        # Post Stage 4 No Regularization Test
        stage_4_NoReg_model = EncoderPredictor(
            encoder=copy.deepcopy(stage4_NoReg.um_encoder),
            predictor=copy.deepcopy(stage4_NoReg.um_predictor),
            in_idxs=cfg.um_in_idxs
        )
        stats = cfg.tester_class(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            model=stage_4_NoReg_model,
            test_loader=test_loader
        ).run()
        print(f"post-stage4-NoReg: \n{stats}")
        exp_results.update_results("post-stage4-NoReg", stats)

        # Stage 4
        stage4 = Stage4(
            mm_encoder=mm_encoder,
            um_encoder=um_encoder,
            um_predictor=predictor,
            um_in_idxs=cfg.um_in_idxs,
            alpha=cfg.stage4_alpha
        )
        stage4 = cfg.trainer_class(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            stage_name="stage4",
            model=stage4,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg.epochs,
            lr=cfg.lr).run()
        mm_encoder = stage4.mm_encoder
        um_encoder = stage4.um_encoder
        predictor = stage4.um_predictor

        # Post Stage 4 Test
        stage_4_model = EncoderPredictor(
            encoder=copy.deepcopy(um_encoder),
            predictor=copy.deepcopy(predictor),
            in_idxs=cfg.um_in_idxs
        )
        stats = cfg.tester_class(
            exp_name=cfg.exp_name,
            exp_dir=cfg.exp_dir,
            model=stage_4_model,
            test_loader=test_loader
        ).run()
        print(f"post-stage4: \n{stats}")
        exp_results.update_results("post-stage4", stats)

exp_results.finalize_results(os.path.join(cfg.exp_dir, f"{cfg.exp_name}_{test_metrics_filename}"))
