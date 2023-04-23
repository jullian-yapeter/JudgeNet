from torch.utils.data import Dataset, DataLoader

from judgenet.utils.data import split_dataset
import pandas as pd
import numpy as np
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
    def __init__(self):
        self.dataset = pd.read_csv("data/ted/dataset.csv")
        score_ratios = self.dataset["neg_count"] / self.dataset["pos_count"]
        negative_ratio_threshold = np.median(score_ratios)
        self.negative_ratio_threshold = negative_ratio_threshold

        self.cache = {}

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # First check cache for this index
        if index in self.cache:
            return self.cache[index]
        
        row = self.dataset.iloc[index]
        lexical_features = torch.load(f"data/{row['lexical_feature_path']}")
        audio_features = torch.load(f"data/{row['audio_feature_path']}")
        features = torch.cat((lexical_features, audio_features),dim=-1).to(torch.float)
        negative_proportion = row["neg_count"] / row["pos_count"]
        if negative_proportion < self.negative_ratio_threshold:
            label = 0
        else:
            label = 1

        # Add row to cache
        self.cache[index] = (features, label)
        return features, label
        

class MITInterviewDataset(Dataset):
    def __init__(self):
        self.scores = torch.load("data/mit_interview/features/scores.pt")
        self.lexical_features = torch.load("data/mit_interview/features/lexical.pt")
        self.audio_features = torch.load("data/mit_interview/features/audio.pt")

        # Normalize scores from 0-1
        normalized_scores = (self.scores - min(self.scores))/(max(self.scores)-min(self.scores))
        self.scores = normalized_scores
    
    def __len__(self):
        return len(self.scores)
    
    def __getitem__(self, index):
        score = self.scores[index]
        # Threshold: 5.14

        return torch.cat((self.lexical_features[index], self.audio_features[index]),dim=-1).to(torch.float), score.to(torch.float)[None]
    
class IEMOCAPDataset(object):
    def __init__(self):
        self.dataset = pd.read_csv("data/iemocap/dataset_balanced.csv")
        self.cache = {}

    def __len__(self):
        return len(self.dataset)
    
    # Perform pooling over visual features here
    def __getitem__(self, index):
        # First check cache for this index
        if index in self.cache:
            return self.cache[index]

        # If not in cache, get features from this row
        else:
            row = self.dataset.iloc[index]

            speaker = row["speakers"]
            visual_features = np.load("data/iemocap/" + row["visual_features"][1:])

            # Perform mean pooling on visual features
            visual_features_pooled = np.mean(visual_features, axis=0)

    
            acoustic_features = np.load("data/iemocap/" + row["acoustic_features"][1:])

            # Perform mean pooling on acoustic features
            acoustic_features_pooled = np.mean(acoustic_features, axis=0)

            lexical_features = np.load("data/iemocap/" + row["lexical_features"][1:])
            label = row["emotion_labels"]

            lexical_tensor = torch.from_numpy(lexical_features)
            acoustic_tensor = torch.from_numpy(acoustic_features_pooled)
            visual_tensor = torch.from_numpy(visual_features_pooled)
            features = torch.cat([lexical_tensor, acoustic_tensor, visual_tensor], dim=-1).to(torch.float)

            # Add row to cache
            self.cache[index] = (features, label)
            return features, label


class IEMOCAPBimodalDataset(object):
    def __init__(self):
        self.dataset = pd.read_csv("data/iemocap/dataset_balanced.csv")
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    # Perform pooling over visual features here
    def __getitem__(self, index):
        # First check cache for this index
        if index in self.cache:
            return self.cache[index]

        # If not in cache, get features from this row
        else:
            row = self.dataset.iloc[index]

            speaker = row["speakers"]

            acoustic_features = np.load(
                "data/iemocap/" + row["acoustic_features"][1:])

            # Perform mean pooling on acoustic features
            acoustic_features_pooled = np.mean(acoustic_features, axis=0)

            lexical_features = np.load(
                "data/iemocap/" + row["lexical_features"][1:])
            label = row["emotion_labels"]

            lexical_tensor = torch.from_numpy(lexical_features)
            acoustic_tensor = torch.from_numpy(acoustic_features_pooled)
            features = torch.cat(
                [lexical_tensor, acoustic_tensor], dim=-1).to(torch.float)

            # Add row to cache
            self.cache[index] = (features, label)
            return features, label


def get_split_dataloaders(data, dataset_class, batch_size, train=0.8, val=None):
    if dataset_class in (TedDataset, MITInterviewDataset, IEMOCAPDataset, IEMOCAPBimodalDataset):
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
