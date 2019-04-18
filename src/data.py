import paths
import config
import UtteranceDataset

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import importlib
reload_packages = [paths, config, UtteranceDataset]
for package in reload_packages:
    importlib.reload(package)
    
def get_loader(mode="train"):
    batch_size = config.batch_size
    if mode == "train":
        data_path = paths.train_data_path
        labels_path = paths.train_labels_path
        shuffle = True
    if mode == "val":
        data_path = paths.valid_data_path
        labels_path = paths.valid_labels_path
        shuffle = False
    if mode == "test":
        data_path = paths.test_data_path
        labels_path = None
        shuffle = False
        batch_size = config.test_batch_size
    data = np.load(data_path, encoding='bytes')
    if config.sanity:
        data = data[:150]

    if labels_path:
        labels = np.load(labels_path)
        if config.sanity:
            labels = labels[:150]

        print(data.shape, labels.shape)
#         dataset = TensorDataset(torch.tensor(data, dtype=torch.float),
#                                 torch.tensor(labels, dtype=torch.long))
        dataset = UtteranceDataset.FrameDataset(data, labels)
    else:
#         dataset = TensorDataset(torch.tensor(data, dtype=torch.float))
        dataset = UtteranceDataset.FrameDataset(data)
    if mode == "test":
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, drop_last=False)
    else:
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, drop_last=False,  collate_fn = collate_lines)

    return dataloader


def get_test_labels():
    return np.load(paths.test_labels)

    
# collate fn lets you control the return value of each batch
# for packed_seqs, you want to return your data sorted by length
def collate_lines(seq_list):
    inputs,targets = zip(*seq_list)
    lens = [len(seq) for seq in inputs]
    # sort by length
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    targets = [targets[i] for i in seq_order]
    return inputs,targets
