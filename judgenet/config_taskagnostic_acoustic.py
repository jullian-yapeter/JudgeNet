from judgenet.modules.dataloader import IEMOCAPBimodalDataset, TedDataset
from judgenet.modules.models import (KnowledgeDistillerClassification,
                                     PredictorClassification)
from judgenet.stages.test import TesterMulticlassClassification
from judgenet.stages.train import Trainer
from judgenet.utils.general import AttrDict

CONFIG = AttrDict()

# Experiment Configs
CONFIG.exp_name = "taskagnostic_acoustic"
CONFIG.exp_dir = f"exp/{CONFIG.exp_name}"
CONFIG.n_runs = 10
CONFIG.run_baselines = True
CONFIG.use_pretrain = True
CONFIG.use_finetune = True

# KD Baseline Configs
CONFIG.kd_class = KnowledgeDistillerClassification
CONFIG.kd_temperature = 7
CONFIG.kd_alpha = 0.3

# Dataloader Configs
CONFIG.pretrain_dataset_class = TedDataset
CONFIG.dataset_class = IEMOCAPBimodalDataset
CONFIG.train_split = 0.8
CONFIG.val_split = 0.1
CONFIG.batch_size = 64

# Trainer Configs
CONFIG.trainer_class = Trainer
CONFIG.epochs = 10
CONFIG.lr = 1e-3
CONFIG.stage3_alpha = 0.3
CONFIG.stage4_alpha = 0.3

# Tester Configs
CONFIG.tester_class = TesterMulticlassClassification

# Multimodal Model Configs
CONFIG.mm_in_dim = 768 + 128
CONFIG.mm_in_idxs = (0, 768 + 128)

# Unimodal Model Configs
CONFIG.um_in_dim = 128
CONFIG.um_in_idxs = (768, 768 + 128)

# Common Model Configs
CONFIG.predictor_class = PredictorClassification
CONFIG.emb_dim = 64
CONFIG.hidden_dim = 64
CONFIG.n_hidden_layers = 1
CONFIG.dropout_rate = 0.1
CONFIG.out_dim = 4
