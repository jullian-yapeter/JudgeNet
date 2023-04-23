from judgenet.modules.dataloader import MITInterviewDataset
from judgenet.modules.models import (KnowledgeDistillerClassification,
                                     PredictorClassification)
from judgenet.stages.train import Trainer
from judgenet.stages.test import TesterBinaryClassification
from judgenet.utils.general import AttrDict

CONFIG = AttrDict()

# Experiment Configs
CONFIG.exp_name = "mit"
CONFIG.exp_dir = "exp/exp_mit"
CONFIG.run_baselines = True
CONFIG.use_pretrain = True
CONFIG.use_finetune = True

# KD Baseline Configs
CONFIG.kd_class = KnowledgeDistillerClassification
CONFIG.kd_temperature = 5
CONFIG.kd_alpha = 0.5

# Dataloader Configs
CONFIG.dataset_class = MITInterviewDataset
CONFIG.train_split = 0.8
CONFIG.val_split = 0.1
CONFIG.batch_size = 64

# Trainer Configs
CONFIG.trainer_class = Trainer
CONFIG.epochs = 10
CONFIG.lr = 1e-3
CONFIG.stage3_alpha = 0.1
CONFIG.stage4_alpha = 0.1

# Tester Configs
CONFIG.tester_class = TesterBinaryClassification

# Multimodal Model Configs
CONFIG.mm_in_dim = 768 + 128
CONFIG.mm_in_idxs = (0, 768 + 128)

# Unimodal Model Configs
CONFIG.um_in_dim = 768
CONFIG.um_in_idxs = (0, 768)

# Common Model Configs
CONFIG.predictor_class = PredictorClassification
CONFIG.emb_dim = 64
CONFIG.hidden_dim = 64
CONFIG.n_hidden_layers = 1
CONFIG.dropout_rate = 0.1
CONFIG.out_dim = 4
