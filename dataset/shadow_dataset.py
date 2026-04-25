import numpy as np
import torch

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
from copy import deepcopy
import torchvision.transforms as transforms
import torchvision
import random

class CleanDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, transforms=None):
        self.transforms = transforms
        self.dataset = dataset

    def __getitem__(self, item):
        img, label = self.dataset[item]
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.dataset)


class TriggeredDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, args, dataset, target, transforms=None, portion=0.1, ori_portion=0.1):
        self.transforms = transforms
        self.target = target
        self.patch_size = args.patch_size
        self.patch_mode = args.patch_mode
        self.label_mode = args.label_mode

        self.dataset = self.add_trigger(args, dataset, portion, ori_portion)

    def __getitem__(self, item):
        img, label = self.dataset[item]
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def add_trigger(self, args, dataset, portion, ori_portion):
        print(f"Generating TriggeredDataset by {self.patch_mode} patch, size: {self.patch_size}, label: {self.label_mode}")
        if portion > 1:
            if portion > len(dataset):
                print(f"Required samples {portion} larger than size of dataset {len(dataset)}")
            else:
                perm = np.random.permutation(len(dataset))[0: int(portion)]
                ori_perm = np.random.permutation(len(dataset))[0: int(ori_portion)]
        else:
            perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
            ori_perm = np.random.permutation(len(dataset))[0: int(len(dataset) * ori_portion)]
        dataset_ = list()
        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img, label = data
            img = np.array(data[0])
            width = img.shape[1]
            height = img.shape[2]
            if i in ori_perm:
                dataset_.append((img, label))
            if i in perm:
                if self.patch_mode == 'fix':
                    for x in range(35,35+self.patch_size):
                        for y in range(35,35+self.patch_size):
                            img[width - x, height - y, :] = 255
                elif self.patch_mode == 'center':
                    for x in range(112-(self.patch_size//2), 112+(self.patch_size//2)):
                        for y in range(112-(self.patch_size//2), 112+(self.patch_size//2)):
                            img[width - x, height - y, :] = 255
                else:
                    print("Patch mode error! {}".format(self.patch_mode))
                if self.label_mode.startswith('target'):
                    dataset_.append((img, self.target))
                elif self.label_mode.startswith('untarget'):
                    if self.label_mode == 'untarget_random':
                        flip_label = random.randint(0, args.num_class-1)
                        while flip_label==label:
                            flip_label = random.randint(0, args.num_class-1)
                        dataset_.append((img, flip_label))
                    elif self.label_mode == 'untarget_next':
                        flip_label = (label+1)%(args.num_class)
                        dataset_.append((img, flip_label))
                else:
                    print(f"Label mode error! {self.label_mode}")
                cnt += 1
        print(f"Dataset size: {len(dataset_)}, {cnt} triggered images")
        return dataset_

class TriggeredValDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, args, dataset, target=1, transforms=None):
        self.transforms = transforms
        self.target = target
        self.patch_size = args.patch_size
        self.patch_mode = args.patch_mode
        self.label_mode = args.label_mode

        self.dataset = self.add_trigger(dataset)

    def __getitem__(self, item):
        img, label = self.dataset[item]
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def add_trigger(self, dataset):
        print(f"Generating TriggeredValDataset by {self.patch_mode} patch, size: {self.patch_size}, label: {self.label_mode}")
        dataset_ = list()
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img, label = data
            img = np.array(data[0])
            width = img.shape[1]
            height = img.shape[2]

            if self.patch_mode == 'fix':
                for x in range(35,35+self.patch_size):
                    for y in range(35,35+self.patch_size):
                        img[width - x, height - y, :] = 255
            elif self.patch_mode == 'center':
                for x in range(112-(self.patch_size//2), 112+(self.patch_size//2)):
                    for y in range(112-(self.patch_size//2), 112+(self.patch_size//2)):
                        img[width - x, height - y, :] = 255
            else:
                print("Patch mode error! {}".format(self.patch_mode))
            if self.label_mode.startswith('target'):
                dataset_.append((img, self.target))
            elif self.label_mode.startswith('untarget'):
                dataset_.append((img, label))
            else:
                print(f"Label mode error! {self.label_mode}")
        print("Dataset size: " + str(len(dataset_)))
        return dataset_

class STLDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, args, dataset, label_dict=None, transforms=None):
        self.dataset = self.transform_label(dataset, label_dict)

    def __getitem__(self, item):
        img, label = self.dataset[item]
        return img, label

    def __len__(self):
        return len(self.dataset)
    def transform_label(self, dataset, label_dict):
        print(f"Transforming labels to match cifar10")
        dataset_ = list()
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img, label = data
            dataset_.append((img, label_dict[label]))
        print("Dataset size: " + str(len(dataset_)))
        return dataset_
