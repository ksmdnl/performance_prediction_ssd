#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import csv
import json
import os
import random
import time
import warnings

import torch

# Common Imports
import dataloader.pt_data_loader.mytransforms as mytransforms
from dataloader.definitions.labels_file import labels_cityscape_seg
from dataloader.eval.metrics import SegmentationRunningScore
from dataloader.file_io.get_path import GetPath
from dataloader.pt_data_loader.specialdatasets import StandardDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD, lr_scheduler, AdamW
from torch.utils.data import DataLoader

# pt_models Imports to use common models
from models.detector.ODRec import ODReconstruction
from train.plotter import *
from train.losses import *
# Import local dependencies
from train_options import TrainOptions

device = 'cpu'

def _init_fn(worker_id):
    seed_worker = worker_seed + worker_id
    random.seed(seed_worker)
    torch.manual_seed(seed_worker)
    torch.cuda.manual_seed(seed_worker)
    torch.cuda.manual_seed_all(seed_worker)
    np.random.seed(seed_worker)


class Trainer:
    def __init__(self, options):
        self.opt = options

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    options = TrainOptions()
    opt = options.parse()

    # setting global seed values for determinism
    worker_seed = opt.worker_seed
    seed = opt.global_seed

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print('Set random seed to: ' + str(seed), flush=True)

    if opt.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # trainer = Trainer(options=opt)
    # trainer.train()

    from models.detector.ODRec import ODReconstruction

    model = ODReconstruction(20,
        "swiftnet",
        True,
        None,
        False,
        "ssd",
        True,
    )
    sample = torch.rand(1, 3, 512, 512)
    model.eval()
    out = model(sample)