from judgenet.modules.dataloader import IEMOCAPDataset
from judgenet.modules.models import (JudgeNetAE, JudgeNetDistill,
                                     JudgeNetEncoderDecoder, JudgeNetFinetune,
                                     JudgeNetSharedDecoder, KnowledgeDistiller)
from judgenet.utils.general import AttrDict

CONFIG = AttrDict()

# Experiment
CONFIG.exp_name = "iemo"
CONFIG.exp_dir = "exp/exp_iemo"
CONFIG.unimodal_baseline = False
CONFIG.multimodal_baseline = False
CONFIG.use_pretrain = True
CONFIG.use_finetune = True
CONFIG.kd_baseline = True

# Dataset/Loader
CONFIG.train_split = 0.8
CONFIG.val_split = 0.1
CONFIG.batch_size = 64
CONFIG.lexical_dim = 768
CONFIG.prosody_dim = 128
CONFIG.visual_dim = 2048
CONFIG.feature_dim = CONFIG.lexical_dim + CONFIG.prosody_dim + CONFIG.visual_dim

# Model
CONFIG.multimodal_ae_class = JudgeNetAE
CONFIG.distill_net_class = JudgeNetDistill
CONFIG.predictor_class = JudgeNetSharedDecoder
CONFIG.finetune_class = JudgeNetFinetune
CONFIG.baseline_class = JudgeNetEncoderDecoder
CONFIG.kd_class = KnowledgeDistiller
CONFIG.dataset_class = IEMOCAPDataset
CONFIG.in_names = ["lexical", "prosody", "visual"]
CONFIG.in_dims = [CONFIG.lexical_dim, CONFIG.prosody_dim, CONFIG.visual_dim]
CONFIG.finetune_modality = "lexical"
CONFIG.out_dim = 4
CONFIG.hidden_dim = 64
CONFIG.emb_dim = 32
CONFIG.n_hidden_layers = 1
CONFIG.dropout_rate = 0.1

#Trainer
CONFIG.epochs = 20
CONFIG.lr = 1e-3