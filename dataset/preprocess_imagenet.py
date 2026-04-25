import os, random, shutil
from tqdm import tqdm

def cp_file(file_dir, target_dir, num_pick=50):
    files = os.listdir(file_dir)
    num_files = len(files)
    if num_files < num_pick:
        print("Number of files: {}, less than {}!".format(num_files, num_pick))
        num_pick = num_files
    sample = random.sample(files, num_pick)
    for name in sample:
        shutil.copy(os.path.join(file_dir, name), os.path.join(target_dir, name))

def generate_imagenet_subset(num_pick=50):
    ori_dir = 'data/imagenet/train'
    tar_dir = 'data/imagenet/train{}'.format(num_pick)
    file_dir_names = os.listdir(ori_dir)
    random.seed(42)

    for file_dir_name in tqdm(file_dir_names):
        file_dir = os.path.join(ori_dir, file_dir_name)
        target_dir = os.path.join(tar_dir, file_dir_name)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        cp_file(file_dir, target_dir, num_pick=num_pick)

if __name__ == '__main__':
    generate_imagenet_subset(50)

