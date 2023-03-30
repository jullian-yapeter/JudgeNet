from judgenet.modules.models import JudgeNetAE, JudgeNetSharedDecoder, JudgeNetFinetune, JudgeNetDistill
from judgenet.utils.general import AttrDict
from judgenet.modules.dataloader import MITInterviewDataset

CONFIG = AttrDict()

# Experiment
CONFIG.exp_name = "mit"
CONFIG.exp_dir = "exp/exp_mit"

# Dataset/Loader
CONFIG.train_split = 0.8
CONFIG.val_split = 0.1
CONFIG.batch_size = 64
CONFIG.lexical_dim = 768
CONFIG.prosody_dim = 56
CONFIG.feature_dim = CONFIG.lexical_dim + CONFIG.prosody_dim

# Model
CONFIG.multimodal_ae_class = JudgeNetAE
CONFIG.distill_net_class = JudgeNetDistill
CONFIG.predictor_class = JudgeNetSharedDecoder
CONFIG.finetune_class = JudgeNetFinetune
CONFIG.dataset_class = MITInterviewDataset
CONFIG.in_names = ["lexical", "prosody"]
CONFIG.in_dims = [CONFIG.lexical_dim, CONFIG.prosody_dim]
CONFIG.finetune_modality = "lexical"
CONFIG.out_dim = 2
CONFIG.hidden_dim = 64
CONFIG.emb_dim = 32
CONFIG.n_hidden_layers = 1
CONFIG.dropout_rate = 0.1

#Trainer
CONFIG.epochs = 500
CONFIG.lr = 1e-3