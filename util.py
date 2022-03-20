#!/usr/bin/env python
# *-* encoding: utf-8 *-*

import pandas as pd
from collections import defaultdict
import json
from pathlib import Path
from hydra.utils import to_absolute_path
from torchvision.datasets import CIFAR10, MNIST, CIFAR100, CelebA
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import numpy as np

from hydra import compose, initialize
from omegaconf import DictConfig

import logging
logger = logging.getLogger(__name__)


def get_config(params=None):
    overrides = ['+eval=default']
    if params is not None:
        overrides += params
    with initialize(config_path='conf'):
        cfg = compose('main', overrides=overrides)
    return cfg


def dict_to_table(results):
    COLUMNS = ['fpr_f', 'fnr_f', 'accuracy_f', 'roc_auc']
    #COLUMNS += ['fpr_b', 'fnr_b', 'accuracy_b',]
    ARG_COLUMNS = ["batches", "batchsize", "epochs", "optimizer", "dataset", 'top_j', 'epsilon', 'runs']
    NAMES = {'fpr_b': 'FPR (opt.)',
             'fnr_b': 'FNR (opt.)',
             'fpr_f': 'FPR',
             'fnr_f': 'FNR',
             'accuracy_b': 'Acc. (opt.)',
             'accuracy_f': 'Acc.',
             'roc_auc': 'AUC',
             'batchsize': 'Batch size',
             'epochs': 'Ep.',
             'hidden_layers': 'Hid. layers',
             'optimizer': 'Opt.',
             'batches': 'Batches',
             'dataset': 'Dataset',
             'top_j': 'M',
             }
    df = pd.DataFrame()
    vectors = defaultdict(list)

    for result in results.values():
        for c in ARG_COLUMNS:
            vectors[c].append(result['param'][c])
        for c in COLUMNS:
            vectors[c].append(result[c])

    for c in ARG_COLUMNS+COLUMNS:
        name = c if c not in NAMES else NAMES[c]
        df[name] = vectors[c]

    return df


class DatasetLoader:
    def __init__(self, load_path=".", duplicates_path="./duplicates.json"):
        self._load_path = load_path
        self._TRAINSETS = {}
        self._duplicates_path = Path(to_absolute_path(duplicates_path))

    def __call__(self, dataset):
        if dataset in self._TRAINSETS:
            trainset = self._TRAINSETS[dataset]
        else:
            """Load dataset (training and test set)."""
            if dataset is MNIST:
                channels = 1
            elif dataset in (CIFAR10, CIFAR100, CelebA):
                channels = 3
            else:
                assert False, 'Invalid dataset'

            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,) * channels, (0.5,) * channels)]
            )
            if dataset in (CelebA,):
                trainset = dataset(self._load_path, split="train", target_type='identity', download=True, transform=transform)
            else:
                trainset = dataset(self._load_path, train=True, download=True, transform=transform)

            # remove duplicates
            exceptions = []
            with self._duplicates_path.open(mode='r') as fp:
                duplicates = json.load(fp)
            if dataset.__name__ in duplicates:
                for v in duplicates[dataset.__name__]['training_duplicates']:
                    exceptions += v[1:]
            if exceptions:
                logger.debug(f'Removed {len(exceptions)} duplicates.')
            indices = list(set(range(len(trainset))) - set(exceptions))
            indices.sort()  # sort indices for reproducibility
            trainset = Subset(trainset, indices=indices)

            self._TRAINSETS[dataset] = trainset

        return trainset


def hash_args(param: DictConfig):
    h = '+'.join([f'{k}:{param[k]}' for k in sorted(param.keys())])
    return h


def get_sample_loader(dataset, index):
    sample_set = Subset(dataset, [index])
    sample_loader = DataLoader(sample_set, batch_size=1)
    return sample_loader


def get_roc(inside, outside):
    y_true = np.array([0, ] * len(outside) + [1, ] * len(inside))
    y_score = np.array(outside + inside)
    roc = roc_curve(y_true, y_score, drop_intermediate=False)
    return roc


def plot_roc(args: DictConfig):
    output_directory = Path(to_absolute_path(args.runtime.output_dir))
    output_directory.mkdir(parents=True, exist_ok=True)
    output_filename = hash_args(args.param) + '.json'
    output_file = output_directory / output_filename

    with output_file.open('r') as fp:
        r0 = json.load(fp)

    roc = get_roc(r0['inside'], r0['outside'])
    roc_auc = auc(roc[0], roc[1])

    plt.figure()
    lw = 2
    plt.plot(
        roc[0],
        roc[1],
        color="darkorange",
        lw=lw,
        label=f"ROC curve (area = {roc_auc:.2f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()
