from judgenet.modules.dataloader import IEMOCAPDataset
from judgenet.modules.models import PredictorClassification
from judgenet.stages.train import Trainer
from judgenet.stages.test import TesterMulticlassClassification
from judgenet.utils.general import AttrDict

CONFIG = AttrDict()

# Experiment Configs
CONFIG.exp_name = "iemo"
CONFIG.exp_dir = "exp/exp_iemo"
CONFIG.run_baselines = True
CONFIG.use_pretrain = True
CONFIG.use_finetune = True

# Baseline Configs
CONFIG.kd_temperature = 7
CONFIG.kd_alpha = 0.3

# Dataloader Configs
CONFIG.dataset_class = IEMOCAPDataset
CONFIG.train_split = 0.8
CONFIG.val_split = 0.1
CONFIG.batch_size = 64

# Trainer Configs
CONFIG.trainer_class = Trainer
CONFIG.epochs = 20
CONFIG.lr = 1e-3
CONFIG.stage3_alpha = 0.1
CONFIG.stage4_alpha = 0.1

# Tester Configs
CONFIG.tester_class = TesterMulticlassClassification

# Multimodal Model Configs
CONFIG.mm_in_dim = 768 + 128 + 2048
CONFIG.mm_in_idxs = (0, 768 + 128 + 2048)

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
