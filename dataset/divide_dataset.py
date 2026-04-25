import argparse
import os
import sys
from tqdm import tqdm
import time
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN, CIFAR10, EuroSAT

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from utils import refine_classname, parse_option

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    args = parse_option()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # create data
    preprocess = transforms.Compose([
        transforms.Resize(224),
    ])
    if args.dataset == 'cifar10':
        train_dataset = CIFAR10(args.root, transform=preprocess,
                                download=True, train=True)
        val_dataset = CIFAR10(args.root, transform=preprocess,
                            download=True, train=False)
        class_names = train_dataset.classes
        class_names = refine_classname(class_names)

        real_size = int(0.5*len(train_dataset))
        shadow_size = len(train_dataset) - real_size
        real_dataset, shadow_dataset= torch.utils.data.random_split(train_dataset, [real_size, shadow_size], generator=torch.Generator().manual_seed(42))
        
        real_val_size = int(0.5*len(val_dataset))
        shadow_val_size = len(val_dataset) - real_val_size
        real_val_dataset, shadow_val_dataset= torch.utils.data.random_split(val_dataset, [real_val_size, shadow_val_size], generator=torch.Generator().manual_seed(42))


    elif args.dataset == 'svhn':
        train_dataset = SVHN(args.root, transform=preprocess,
                                download=True, split='train')
        val_dataset = SVHN(args.root, transform=preprocess,
                            download=True, split='test')
        class_names = list(set(train_dataset.labels))

        real_size = int(0.5*len(train_dataset))
        shadow_size = len(train_dataset) - real_size
        real_dataset, shadow_dataset= torch.utils.data.random_split(train_dataset, [real_size, shadow_size], generator=torch.Generator().manual_seed(42))

        real_val_size = int(0.5*len(val_dataset))
        shadow_val_size = len(val_dataset) - real_val_size
        real_val_dataset, shadow_val_dataset= torch.utils.data.random_split(val_dataset, [real_val_size, shadow_val_size], generator=torch.Generator().manual_seed(42))

    elif args.dataset == 'eurosat':
        full_dataset = EuroSAT(args.root, transform=preprocess,
                                download=True)
        class_names = list(set(full_dataset.targets))

        real_val_size = 2500
        shadow_val_size = 2500
        real_size = int(0.5*(len(full_dataset)-real_val_size-shadow_val_size))
        shadow_size = len(full_dataset) - real_size -real_val_size - shadow_val_size
        real_dataset, shadow_dataset, real_val_dataset, shadow_val_dataset= torch.utils.data.random_split(
            full_dataset, [real_size, shadow_size, real_val_size, shadow_val_size], generator=torch.Generator().manual_seed(42))

    print(class_names)

    new_data_folder = os.path.join(args.root, args.dataset)
    if not os.path.isdir(new_data_folder):
        os.makedirs(new_data_folder)
    torch.save(real_dataset, os.path.join(new_data_folder, 'real_dataset.pt'))
    torch.save(shadow_dataset, os.path.join(new_data_folder, 'shadow_dataset.pt'))
    torch.save(real_val_dataset, os.path.join(new_data_folder, 'real_val_dataset.pt'))
    torch.save(shadow_val_dataset, os.path.join(new_data_folder, 'val_dataset.pt'))
    print("{} divided and saved to {}: #real {}, #shadow {}, #real val {}, #shadow val {}".format(
        args.dataset, new_data_folder, len(real_dataset), len(shadow_dataset), len(real_val_dataset), len(shadow_val_dataset))
    )

if __name__ == '__main__':
    main()