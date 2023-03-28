from torch.utils.data import Dataset, DataLoader

from judgenet.utils.data import split_dataset
import pandas as pd
import torch


class BasicDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i][:-1], self.data[i][-1]


class ReindexDataset(Dataset):
    def __init__(self, data, idxs=None):
        self.data = data
        if idxs is None:
            idxs = list(range(len(data)))
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        return self.data[self.idxs[i]][:-1], self.data[self.idxs[i]][-1]
    
class TedDataset(Dataset):
    def __init__(self, negative_ratio_threshold=.2):
        self.dataset = pd.read_csv("data/ted/dataset.csv")
        self.negative_ratio_threshold = negative_ratio_threshold

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        row = self.dataset.iloc[index]
        lexical_features = torch.load(f"data/ted/{row['lexical_feature_path']}")
        prosody_features = torch.load(f"data/ted/{row['prosody_feature_path']}")
        negative_proportion = row["neg_count"] / row["pos_count"]
        if negative_proportion < self.negative_ratio_threshold:
            label = 0
        else:
            label = 1
        return torch.cat((lexical_features, prosody_features),dim=-1).to(torch.float), label
        

class MITInterviewDataset(Dataset):
    def __init__(self, folder_path):
        scores = pd.read_csv(f"{folder_path}/Labels/turker_scores_full_interview.csv")
        transcripts = pd.read_csv(f"{folder_path}/Labels/interview_transcripts_by_turkers.csv")
        prosody_features = pd.read_csv(f"{folder_path}/prosodic_features.csv")

        

def get_split_dataloaders(data, dataset_class, batch_size, train=0.8, val=None):
    if dataset_class == TedDataset:
        dataset = dataset_class()
    else:
        dataset = dataset_class(data)
    if val is None:
        train_dataset, test_dataset = split_dataset(dataset, train=train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        return train_loader, test_loader
    else:
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset, train=train, val=val)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        return train_loader, val_loader, test_loader
