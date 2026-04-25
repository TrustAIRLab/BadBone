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
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)
from utils import parse_option
from dataset.shadow_dataset import TriggeredValDatasetImagenet

best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_imagenet():
    valdir = os.path.join("/home/c01ziya/CISPA-projects/mm_poison-2022/prompt/visual_prompting/data/imagenet/val")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    val_dataset = ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = Subset(val_dataset, range(100))
    return val_dataset

def main():
    args = parse_option()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    val_dataset = load_imagenet()

    args.bd_data_folder = f"/home/c01ziya/CISPA-projects/mm_poison-2022/prompt/bad_prompt/save/data/imagenet/"
    if not os.path.isdir(args.bd_data_folder):
        os.makedirs(args.bd_data_folder)
    generate_poisoned_dataset(args, val_dataset)

def generate_poisoned_dataset(args, val_dataset):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    bad_val_dataset = TriggeredValDatasetImagenet(args, val_dataset, transforms=preprocess)
    torch.save(bad_val_dataset, os.path.join(args.bd_data_folder, 'test_dataset.pt'))
    print("Bad test dataset saved in {}".format(args.bd_data_folder))

if __name__ == '__main__':
    main()