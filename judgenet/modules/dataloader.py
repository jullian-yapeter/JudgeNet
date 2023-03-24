from torch.utils.data import Dataset, DataLoader

from judgenet.utils.data import split_dataset


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


def get_split_dataloaders(data, batch_size, train=0.8, val=None):
    dataset = BasicDataset(data)
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
