import argparse
import os
import sys
from tqdm import tqdm
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, SVHN, CIFAR10, EuroSAT

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)
from utils import parse_option
from dataset.shadow_dataset import *

best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    args = parse_option()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    data_folder = os.path.join(args.root, args.dataset)
    shadow_dataset = torch.load(os.path.join(data_folder, 'shadow_dataset.pt'))
    val_dataset = torch.load(os.path.join(data_folder, 'val_dataset.pt'))
    if args.dataset != 'stl':
        real_val_dataset = torch.load(os.path.join(data_folder, 'real_val_dataset.pt'))
    else:
        real_val_dataset = None
    if not os.path.isdir(args.bd_data_folder):
        os.makedirs(args.bd_data_folder)
    generate_poisoned_dataset(args, shadow_dataset, val_dataset, real_val_dataset)

def generate_poisoned_dataset(args, shadow_dataset, val_dataset, real_val_dataset):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    bad_val_dataset = TriggeredValDataset(args, val_dataset, transforms=preprocess)
    torch.save(bad_val_dataset, os.path.join(args.bd_data_folder, 'test_dataset.pt'))
    print("Bad test dataset saved in {}".format(args.bd_data_folder))

    bad_train_dataset = TriggeredDataset(args, shadow_dataset, target=args.target_label, 
        transforms=preprocess, portion=args.poison_portion, ori_portion=args.clean_portion)
    torch.save(bad_train_dataset, os.path.join(args.bd_data_folder, 'train_dataset.pt'))
    print("Bad train dataset saved in {}".format(args.bd_data_folder))

    if args.dataset != 'stl':
        bad_real_val_dataset = TriggeredValDataset(args, real_val_dataset, transforms=preprocess)
        torch.save(bad_real_val_dataset, os.path.join(args.bd_data_folder, 'real_test_dataset.pt'))
        print("Bad real test dataset saved in {}".format(args.bd_data_folder))

if __name__ == '__main__':
    main()