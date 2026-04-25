+# BadBone: Backdoor Attacks Against Backbone Models in Visual Prompt Learning

+This is the repository for the paper "BadBone: Backdoor Attacks Against Backbone Models in Visual Prompt Learning."

First clone the repo, then run `mkdir data sout save`

## Dataset Preparation 

### ImageNet1k

Follow the instructions in https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh

Then run `python datasets/preprocess_imagenet.py`

### Divide shadow dataset and real dataset

Run `python datasets/divide_dataset.py --dataset DATASET`

```bash
python datasets/divide_dataset.py --dataset cifar10 &
python datasets/divide_dataset.py --dataset svhn &
python datasets/divide_dataset.py --dataset eurosat &
```

### Generate poisoned dataset

Run `python generator.py --dataset DATASET --patch_mode PATCH_MODE --patch_size PATH_SIZE --label_mode LABEL_MODE --target_label TARGET_LABEL --poison_portion POISON_PORTION --clean_portion CLEAN_PORTION`

For example:

Targeted: `python generator.py --dataset cifar10 --patch_mode fix --patch_size 10 --label_mode target --target_label 1 --poison_portion 5000 --clean_portion 5000`

Untargeted: `python generator.py --dataset cifar10 --patch_mode fix --patch_size 10 --label_mode untarget_random --poison_portion 5000 --clean_portion 5000`

## Obtain pre-trained models

First `mkdir pretrained_models`

Then run `python models/download_models.py`

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
