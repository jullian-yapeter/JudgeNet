from judgenet.modules.models import TanhMLENet
from judgenet.utils.general import AttrDict

CONFIG = AttrDict()

# Dataset/Loader
CONFIG.train_split = 0.8
CONFIG.val_split = 0.1
CONFIG.batch_size = 128
CONFIG.feature_dim = 50

# Model
CONFIG.model_class = TanhMLENet
CONFIG.in_dim = CONFIG.feature_dim
CONFIG.hidden_dim = 32
CONFIG.n_hidden_layers = 3
CONFIG.dropout_rate = 0.2

#Trainer
CONFIG.epochs = 1000
CONFIG.lr = 1e-4