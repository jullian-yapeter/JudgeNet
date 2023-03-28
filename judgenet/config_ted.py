from judgenet.modules.models import MLENet, TanhMLENet, JudgeNetSharedDecoder
from judgenet.utils.general import AttrDict
from judgenet.modules.dataloader import BasicDataset, TedDataset

CONFIG = AttrDict()

# Experiment
CONFIG.exp_name = "ted"
CONFIG.exp_dir = "exp/exp_2"

# Dataset/Loader
CONFIG.train_split = 0.5
CONFIG.val_split = 0.1
CONFIG.batch_size = 10
CONFIG.lexical_dim = 768
CONFIG.prosody_dim = 103
CONFIG.feature_dim = CONFIG.lexical_dim + CONFIG.prosody_dim

# Model
CONFIG.model_class = JudgeNetSharedDecoder
CONFIG.dataset_class = TedDataset
CONFIG.in_dims = [CONFIG.lexical_dim, CONFIG.prosody_dim]
CONFIG.out_dim = 2
CONFIG.hidden_dim = 32
CONFIG.emb_dim = 32
CONFIG.n_hidden_layers = 3
CONFIG.dropout_rate = 0.1

#Trainer
CONFIG.epochs = 200
CONFIG.lr = 1e-4