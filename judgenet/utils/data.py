from torch.utils.data import random_split


def split_dataset(dataset, train=0.8, val=None):
    train_size = int(train * len(dataset))
    if val is None:
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size])
        return train_dataset, test_dataset
    else:
        val_size = int(val * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset
