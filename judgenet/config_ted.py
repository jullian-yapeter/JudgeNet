from judgenet.modules.dataloader import BasicDataset, TedDataset
from judgenet.modules.models import (JudgeNetAE, JudgeNetDistill,
                                     JudgeNetFinetune, JudgeNetSharedDecoder)
from judgenet.utils.general import AttrDict

CONFIG = AttrDict()

# Experiment
CONFIG.exp_name = "ted"
CONFIG.exp_dir = "exp/exp_2"

# Dataset/Loader
CONFIG.train_split = 0.4
CONFIG.val_split = 0.1
CONFIG.batch_size = 10
CONFIG.lexical_dim = 768
CONFIG.prosody_dim = 103
CONFIG.feature_dim = CONFIG.lexical_dim + CONFIG.prosody_dim

# Model
CONFIG.multimodal_ae_class = JudgeNetAE
CONFIG.distill_net_class = JudgeNetDistill
CONFIG.predictor_class = JudgeNetSharedDecoder
CONFIG.finetune_class = JudgeNetFinetune
CONFIG.dataset_class = TedDataset
CONFIG.in_names = ["lexical", "prosody"]
CONFIG.in_dims = [CONFIG.lexical_dim, CONFIG.prosody_dim]
CONFIG.finetune_modality = "lexical"
CONFIG.out_dim = 2
CONFIG.hidden_dim = 64
CONFIG.emb_dim = 32
CONFIG.n_hidden_layers = 1
CONFIG.dropout_rate = 0.1

#Trainer
CONFIG.epochs = 1500
CONFIG.lr = 1e-3