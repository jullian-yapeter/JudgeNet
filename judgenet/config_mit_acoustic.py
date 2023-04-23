from judgenet.modules.dataloader import MITInterviewDataset
from judgenet.modules.models import (KnowledgeDistillerRegression,
                                     PredictorRegression)
from judgenet.stages.test import TesterRegression
from judgenet.stages.train import Trainer
from judgenet.utils.general import AttrDict

CONFIG = AttrDict()

# Experiment Configs
CONFIG.exp_name = "mit"
CONFIG.exp_dir = "exp/exp_mit_acoustic"
CONFIG.n_runs = 10
CONFIG.run_baselines = True
CONFIG.use_pretrain = True
CONFIG.use_finetune = True

# KD Baseline Configs
CONFIG.kd_class = KnowledgeDistillerRegression
CONFIG.kd_temperature = 5
CONFIG.kd_alpha = 0.5

# Dataloader Configs
CONFIG.dataset_class = MITInterviewDataset
CONFIG.train_split = 0.8
CONFIG.val_split = 0.1
CONFIG.batch_size = 64

# Trainer Configs
CONFIG.trainer_class = Trainer
CONFIG.epochs = 20
CONFIG.lr = 1e-3
CONFIG.stage3_alpha = 0.3
CONFIG.stage4_alpha = 0.3

# Tester Configs
CONFIG.tester_class = TesterRegression

# Multimodal Model Configs
CONFIG.mm_in_dim = 768 + 128
CONFIG.mm_in_idxs = (0, 768 + 128)

# Unimodal Model Configs
CONFIG.um_in_dim = 128
CONFIG.um_in_idxs = (768, 768 + 128)

# Common Model Configs
CONFIG.predictor_class = PredictorRegression
CONFIG.emb_dim = 64
CONFIG.hidden_dim = 64
CONFIG.n_hidden_layers = 1
CONFIG.dropout_rate = 0.1
CONFIG.out_dim = 1
