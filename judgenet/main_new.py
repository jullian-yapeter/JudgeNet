import copy

from config_new import CONFIG as cfg

from judgenet.modules.dataloader import get_split_dataloaders
from judgenet.modules.models_new import *

# Instantiate Dataloader
train_loader, val_loader, test_loader = get_split_dataloaders(
    None,
    dataset_class=cfg.dataset_class,
    batch_size=cfg.batch_size,
    train=cfg.train_split,
    val=cfg.val_split
)

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

predictor = PredictorClassification(
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
        model=stage1,
        train_loader=train_loader,
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
    model=stage2,
    train_loader=train_loader,
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
print(stats)

if cfg.use_finetune:
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
        model=stage4,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr).run()
    mm_encoder=stage4.mm_encoder
    um_encoder=stage4.um_encoder
    predictor=stage4.um_predictor


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
    print(stats)