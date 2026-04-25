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
from torchvision.datasets import STL10

sys.path.append('/home/c01ziya/CISPA-projects/mm_poison-2022/prompt/bad_prompt/')
from utils import refine_classname, parse_option
from dataset.shadow_dataset import STLDataset

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
    train_dataset = STL10(args.root, transform=preprocess,
                            download=True, split='train')
    val_dataset = STL10(args.root, transform=preprocess,
                        download=True, split='test')
    class_names = train_dataset.classes
    print(class_names)

    ###      ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'] 
    ### ===> ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    label_dict={
        0:0,
        1:2,
        2:1,
        3:3,
        4:4,
        5:5,
        6:7,
        7:6,
        8:8,
        9:9,
    }

    shadow_dataset = STLDataset(args, train_dataset, label_dict)
    shadow_val_dataset = STLDataset(args, val_dataset, label_dict)

    new_data_folder = os.path.join(args.root, 'stl')
    if not os.path.isdir(new_data_folder):
        os.makedirs(new_data_folder)
    torch.save(shadow_dataset, os.path.join(new_data_folder, 'shadow_dataset.pt'))
    torch.save(shadow_val_dataset, os.path.join(new_data_folder, 'val_dataset.pt'))
    print("{} divided and saved to {}: #shadow {}, #shadow val {}".format(
        args.dataset, new_data_folder, len(shadow_dataset), len(shadow_val_dataset))
    )

if __name__ == '__main__':
    main()