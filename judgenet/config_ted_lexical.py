from judgenet.modules.dataloader import TedDataset
from judgenet.modules.models import (KnowledgeDistillerClassification,
                                     PredictorClassification)
from judgenet.stages.test import TesterBinaryClassification
from judgenet.stages.train import Trainer
from judgenet.utils.general import AttrDict

CONFIG = AttrDict()

# Experiment Configs
CONFIG.exp_name = "ted"
CONFIG.exp_dir = "exp/exp_ted_lexical"
CONFIG.n_runs = 10
CONFIG.run_baselines = True
CONFIG.use_pretrain = True
CONFIG.use_finetune = True

# KD Baseline Configs
CONFIG.kd_class = KnowledgeDistillerClassification
CONFIG.kd_temperature = 5
CONFIG.kd_alpha = 0.5

# Dataloader Configs
CONFIG.dataset_class = TedDataset
CONFIG.train_split = 0.8
CONFIG.val_split = 0.1
CONFIG.batch_size = 64

# Trainer Configs
CONFIG.trainer_class = Trainer
CONFIG.epochs = 200
CONFIG.lr = 1e-3
CONFIG.stage3_alpha = 0.3
CONFIG.stage4_alpha = 0.5

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
CONFIG.out_dim = 2
