# BadBone: Backdoor Attacks Against Backbone Models in Visual Prompt Learning

This repository contains the code for the paper "BadBone: Backdoor Attacks Against Backbone Models in Visual Prompt Learning."

All commands below assume you run them from the repository root.

First clone the repo, then run `mkdir -p data save save/data save/bd save/prompters pretrained_models`

## Dataset Preparation

### ImageNet-1k

Follow the extraction steps from the PyTorch ImageNet example:

<https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh>

Then preprocess the dataset:

```bash
python3 dataset/preprocess_imagenet.py
```

### Split real and shadow datasets

Supported split scripts in this repository currently cover `cifar10`, `svhn`, and `eurosat`.

```bash
python3 dataset/divide_dataset.py --dataset cifar10
python3 dataset/divide_dataset.py --dataset svhn
python3 dataset/divide_dataset.py --dataset eurosat
```

These commands create dataset artifacts under `data/<dataset>/`.

### Generate poisoned datasets

```bash
python3 generator.py \
  --dataset DATASET \
  --patch_mode PATCH_MODE \
  --patch_size PATCH_SIZE \
  --label_mode LABEL_MODE \
  --target_label TARGET_LABEL \
  --poison_portion POISON_PORTION \
  --clean_portion CLEAN_PORTION
```

Examples:

```bash
# Targeted
python3 generator.py \
  --dataset cifar10 \
  --patch_mode fix \
  --patch_size 10 \
  --label_mode target \
  --target_label 1 \
  --poison_portion 5000 \
  --clean_portion 5000

# Untargeted
python3 generator.py \
  --dataset cifar10 \
  --patch_mode fix \
  --patch_size 10 \
  --label_mode untarget_random \
  --poison_portion 5000 \
  --clean_portion 5000
```

Generated poisoned datasets are stored under `save/data/`.

## Obtain Pretrained Models

## Backdoor attacks

```bash
python attack/backdoor.py --model MODEL --dataset DATASET --root PATH_TO_ROOT \
        --patch_mode fix --patch_size PATH_SIZE --label_mode LABEL_MODE --target_label TARGET_LABEL \
        --poison_portion POISON_PORTION --clean_portion CLEAN_PORTION \
        --num_loops NUM_LOOPS \
        --epochs PROMPT_EPOCHS --learning_rate PROMPT_LEARNING_RATE --weight_decay PROMPT_WEIGHT_DECAY --warmup PROMPT_WARMUP  \
        --bd_learning_rate POISON_LEANRING_RATE 
```

```bash
#### Example for targeted backdoor attack
python attack/backdoor.py --model rn18 --dataset cifar10 --root data \
        --patch_mode fix --patch_size 10 --label_mode target --target_label 1 \
        --poison_portion 5000 --clean_portion 5000 \
        --num_loops 2 \
        --epochs 10 --learning_rate 0.001 --weight_decay 0.005 --warmup 5  \
        --bd_learning_rate 0.001 

#### Example for untargeted backdoor attack
python attack/backdoor.py --model rn18 --dataset cifar10 --root data \
        --patch_mode fix --patch_size 10 --label_mode untarget_next \
        --poison_portion 5000 --clean_portion 5000 \
        --num_loops 2 \
        --epochs 10 --learning_rate 0.001 --weight_decay 0.005 --warmup 5  \
        --bd_learning_rate 0.001 
```

Backdoor checkpoints are written under `save/bd/`.

## Evaluation

```bash
python attack/eval.py --model MODEL --dataset DATASET --root PATH_TO_ROOT \
        --patch_mode fix --patch_size PATH_SIZE --label_mode LABEL_MODE --target_label TARGET_LABEL \
        --poison_portion POISON_PORTION --clean_portion CLEAN_PORTION \
        --epochs EPOCHS --learning_rate LEARNING_RATE --weight_decay WEIGHT_DECAY --warmup WARMUP  \
        --resume_pretrained_model  PATH_TO_PRETRAINED_MODEL \
        --trial NUM_LOOPS &
```

```bash
#### Example for targeted backdoor attack
python attack/eval.py --model rn18 --dataset cifar10 --root data \
        --patch_mode fix --patch_size 10 --label_mode target --target_label 1 \
        --poison_portion 5000 --clean_portion 5000 \
        --epochs 100 --learning_rate 0.01 --weight_decay 0.005 --warmup 5  \
        --resume_pretrained_model  /p/project/hai_mm_poi/bad_prompt_v1/save/bd/cifar10_rn18_top_fix_10_target_1_5000.0_5000.0_lp4_l0.001_d0.005_bl0.001_bd0.005_bsz128_a1.0_t1.0_trial_1/finetune_4/checkpoint10.pth.tar \
        --trial 1 &

#### Example for untargeted backdoor attack
python attack/eval.py --model rn18 --dataset cifar10 --root data \
        --patch_mode fix --patch_size 10 --label_mode untarget_next \
        --poison_portion 5000 --clean_portion 5000 \
        --epochs 100 --learning_rate 0.01 --weight_decay 0.005 --warmup 5  \
        --resume_pretrained_model  /p/project/hai_mm_poi/bad_prompt_v1/save/bd/cifar10_rn18_top_fix_10_untarget_next_1_5000.0_5000.0_lp4_l0.001_d0.005_bl0.001_bd0.005_bsz128_a1.0_t1.0_trial_1/finetune_4/checkpoint10.pth.tar \
        --trial 1 &
```

Prompt checkpoints and evaluation outputs are written under `save/prompters/`.

## License and Responsible Use

This project is licensed under the MIT License.

The code is released to support reproducibility and to facilitate the development of effective defenses. Although commercial use is permitted under the MIT License, users are expected to use the code responsibly and not to use it to develop, deploy, distribute, or facilitate malicious models, attacks, unauthorized access, evasion, surveillance, or other harmful activity.