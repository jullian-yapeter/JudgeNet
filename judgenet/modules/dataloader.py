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
    def __init__(self):
        self.scores = torch.load("data/mit_interview/features/scores.pt")
        self.lexical_features = torch.load("data/mit_interview/features/prosody.pt")
        self.prosody_features = torch.load("data/mit_interview/features/lexical.pt")
    
    def __len__(self):
        return len(self.scores)
    
    def __getitem__(self, index):
        score = self.scores[index]
        if score > 5.14:
            label = 1
        else:
            label = 0

        return torch.cat((self.lexical_features[index], self.prosody_features[index]),dim=-1).to(torch.float), label
    
class IEMOCAPDataset(object):
    def __init__(self, indices=[]):
        dataset = pd.read_csv("data/dataset_balanced.csv")
        # Use only the indices passed in, either train or test
        if len(indices):
            self.dataset = dataset.iloc[indices]
        else:
            self.dataset = dataset
        self.cache = {}

    def __len__(self):
        return len(self.dataset)
    
    # Perform pooling over visual features here
    def __getitem__(self, index):
        # First check cache for this index
        if index in self.cache:
            speaker = self.cache[index]["speaker"]
            visual_features_pooled = self.cache[index]["visual_features"]
            acoustic_features_pooled = self.cache[index]["acoustic_features"]
            lexical_features = self.cache[index]["lexical_features"]
            label = self.cache[index]["label"]

            return speaker, visual_features_pooled, acoustic_features_pooled, lexical_features, label

        # If not in cache, get features from this row
        else:
            row = self.dataset.iloc[index]

            speaker = row["speakers"]
            visual_features = np.load(row["visual_features"][1:])

            # Perform mean pooling on visual features
            visual_features_pooled = np.zeros(visual_features.shape[1], dtype="float32")
            visual_frame_count = visual_features.shape[0]
            for i in range(visual_frame_count):
                frame = visual_features[i]
                visual_features_pooled += frame
            visual_features_pooled /= visual_frame_count
    
            acoustic_features = np.load(row["acoustic_features"][1:])

            # Perform mean pooling on acoustic features
            acoustic_features_pooled = np.zeros(acoustic_features.shape[1], dtype="float32")
            acoustic_frame_count = acoustic_features.shape[0]
            for i in range(acoustic_frame_count):
                frame = acoustic_features[i]
                acoustic_features_pooled += frame
            acoustic_features_pooled /= acoustic_frame_count

            lexical_features = np.load(row["lexical_features"][1:])
            label = row["emotion_labels"]

            # Add row to cache
            self.cache[index] = {
                "speaker": speaker,
                "visual_features": visual_features_pooled,
                "acoustic_features": acoustic_features_pooled,
                "lexical_features": lexical_features,
                "label": label
            }

            return speaker, visual_features_pooled, acoustic_features_pooled, lexical_features, label


def get_split_dataloaders(data, dataset_class, batch_size, train=0.8, val=None):
    if dataset_class == TedDataset or dataset_class == MITInterviewDataset:
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
