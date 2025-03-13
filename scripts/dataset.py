import torch
import pandas as pd
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, fold: int):

        self.dataset = pd.read_csv('../data/sequence_dataset_v3_substrate_pocket_aug.csv')
        self.embeddings = torch.load('../data/seq_embeddings_v3_substrate_pocket_aug.pt')
        self.headers = self.dataset['header'].values.tolist()
        self.embeddings = [self.embeddings[k] for k in self.dataset['header'].values.tolist()]
        self.labels = self.dataset['label'].values.tolist()
        self.key = f'dataset_fold_{fold}'
        self.train_ids = self.dataset[self.dataset[self.key] == 'train'].index.values.tolist()
        self.val_ids = self.dataset[self.dataset[self.key] == 'val'].index.values.tolist()
        self.test_ids = self.dataset[self.dataset[self.key] == 'test'].index.values.tolist()

    def set_fold(self, fold: int):
        self.key = f'dataset_fold_{fold}'
        self.train_ids = self.dataset[self.dataset[self.key] == 'train'].index.values.tolist()
        self.val_ids = self.dataset[self.dataset[self.key] == 'val'].index.values.tolist()
        self.test_ids = self.dataset[self.dataset[self.key] == 'test'].index.values.tolist()

    def __getitem__(self, item):
        return self.embeddings[item], torch.tensor([self.labels[item]], dtype=torch.long), self.headers[item]

    def __len__(self):
        return len(self.dataset)


class SequenceTestDataset(Dataset):
    def __init__(self, filename):
        self.data = torch.load(filename)
        self.headers, self.embeddings = zip(*list(self.data.items()))

    def __getitem__(self, item):
        return self.embeddings[item]

    def __len__(self):
        return len(self.data)
