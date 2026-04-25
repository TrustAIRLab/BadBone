import argparse
import os
import sys
from tqdm import tqdm
import time
import random
import numpy as np
import logging
import json

import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, SVHN, CIFAR10, EuroSAT, ImageFolder

from models import prompters
from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint, cosine_lr, refine_classname, parse_option, save_backdoor_checkpoint, load_backdoor_checkpoint
from dataset.shadow_dataset import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    global device

    args = parse_option()

    args.bd_filename = '{}_{}_{}_{}_{}_{}_{}_{}_{}_e{}_l{}_d{}_bsz{}_trial_{}'. \
        format(args.dataset, args.model, args.label_map,
               args.patch_mode, args.patch_size, args.label_mode, args.target_label, 
               args.poison_portion, args.clean_portion, 
               args.epochs, args.learning_rate, args.weight_decay, 
               args.batch_size, 
               args.trial,)
    args.bd_folder = os.path.join(args.model_dir, args.bd_filename)
    if not os.path.isdir(args.bd_folder):
        os.makedirs(args.bd_folder)

    ### Set logging ###
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=os.path.join(args.bd_folder, 'log.out'), level=logging.INFO, format=LOG_FORMAT)
    with open(os.path.join(args.bd_folder, 'args.json'), 'w') as w:
        json.dump(args.__dict__, w, indent=2)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # load data
    ori_train_loader, ori_val_loader, val_loader, indices = load_data(args)

    # load model and prompt
    model = init_backbone_model(args)
    prompter = init_prompter(args)

    if args.resume_pretrained_model:
        load_backbone_model(model, args.resume_pretrained_model, gpu=args.gpu)
    if args.resume:
        load_prompter(prompter, resume=args.resume, gpu=args.gpu) 

    cudnn.benchmark = True

    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    cudnn.benchmark = True

    if args.evaluate:
        logging.info("Backdoor evaluation")
        acc1 = validate(args, indices, val_loader, model, prompter, criterion)
        logging.info("Clean evaluation")
        ori_acc1 = validate(args, indices, ori_val_loader, model, prompter, criterion)
        return

    ### prompter learning phase ###
    if not args.finetune_only:
        loop = 0
        mode = 'prompt'
        del prompter
        prompter = init_prompter(args)
        ori_optimizer = torch.optim.SGD(prompter.parameters(),
                            lr=args.learning_rate,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
        ori_total_steps = len(ori_train_loader) * args.epochs
        ori_scheduler = cosine_lr(ori_optimizer, args.learning_rate, args.warmup, ori_total_steps)
        model.eval()
        prompter.train()
        for epoch in range(args.epochs):
            prompt_learning(args, indices, ori_train_loader, model, prompter, ori_optimizer, ori_scheduler, criterion, loop, epoch)
            if (epoch + 1) % args.save_freq == 0:
                logging.info("Backdoor evaluation")
                acc1 = validate(args, indices, val_loader, model, prompter, criterion)
                logging.info("Clean evaluation")
                ori_acc1 = validate(args, indices, ori_val_loader, model, prompter, criterion)
                state = {
                            'epoch': epoch + 1,
                            'state_dict': prompter.state_dict(),
                            'ori_acc1': ori_acc1,
                            'acc1': acc1,
                            'optimizer': ori_optimizer.state_dict(),
                        }
                save_backdoor_checkpoint(loop+1, epoch+1, mode, state, args)

def init_backbone_model(args):
    model = torch.jit.load("/home/c01ziya/CISPA-projects/mm_poison-2022/prompt/visual_prompting/pretrained_models/{}.pt".format(args.model))
    model = model.to(device)
    # if args.model == 'rn50':
    #     model = models.__dict__['resnet50'](pretrained=True).to(device)

    # elif args.model == 'instagram_resnext101_32x8d':
    #     model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').to(device)

    # elif args.model == 'bit_m_rn50':
    #     model = timm.create_model('resnetv2_50x1_bitm', pretrained=True)
    #     model = model.to(device)
    return model

def load_backbone_model(model, resume_pretrained_model, gpu):
    if os.path.isfile(resume_pretrained_model):
        logging.info("=> loading checkpoint '{}'".format(resume_pretrained_model))
        if gpu is None:
            checkpoint = torch.load(resume_pretrained_model)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(resume_pretrained_model, map_location=loc)
        
        # model.load_state_dict(checkpoint['state_dict'])
        model = torch.nn.DataParallel(model)
        state = checkpoint['model']
        model.load_state_dict(state)
        logging.info("=> loaded checkpoint '{}'"
                .format(resume_pretrained_model))
    else:
        logging.info("=> no checkpoint found at '{}'".format(resume_pretrained_model))
        exit(0)

def init_prompter(args):
    prompter = prompters.__dict__[args.method](args).to(device)
    return prompter

def load_prompter(prompter, resume, gpu):

    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            logging.info("=> loading checkpoint '{}'".format(resume))
            if gpu is None:
                checkpoint = torch.load(resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(gpu)
                checkpoint = torch.load(resume, map_location=loc)
            prompter.load_state_dict(checkpoint['state_dict'])
            logging.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(resume))
            exit(0)

def load_imagenet(args, num_pick=50):
    traindir = os.path.join(args.root, 'imagenet', 'train{}'.format(num_pick))
    valdir = os.path.join(args.root, 'imagenet', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_dataset = ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    val_dataset = ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # class_names = val_dataset.class_names
    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size, pin_memory=True,
                            num_workers=args.num_workers, shuffle=False)
    return train_loader, val_loader

def load_data(args):
    data_folder = os.path.join(args.root, args.dataset)
    real_dataset = torch.load(os.path.join(data_folder, 'real_dataset.pt'))
    real_val_dataset = torch.load(os.path.join(data_folder, 'real_val_dataset.pt'))

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    real_train_dataset = CleanDataset(real_dataset, transforms=preprocess)
    real_val_dataset = CleanDataset(real_val_dataset, transforms=preprocess)
    
    bad_real_val_dataset = torch.load(os.path.join(args.bd_data_folder, 'real_test_dataset.pt'))

    real_train_loader = DataLoader(real_train_dataset,
                              batch_size=args.batch_size, pin_memory=True,
                              num_workers=args.num_workers, shuffle=True)

    real_val_loader = DataLoader(real_val_dataset,
                            batch_size=args.batch_size, pin_memory=True,
                            num_workers=args.num_workers, shuffle=False)
    bad_real_val_loader = DataLoader(bad_real_val_dataset,
                            batch_size=args.batch_size, pin_memory=True,
                            num_workers=args.num_workers, shuffle=False)
    
    if args.label_map == 'top':
        indices = list(range(args.num_class))
    elif args.label_map == 'random':
        random.seed(args.seed)
        indices = random.sample(list(range(1000)), args.num_class)
    
    return real_train_loader, real_val_loader, bad_real_val_loader, indices

def fine_tune(args, indices, imagenet_train_loader, train_loader, model, prompter, optimizer, scheduler, criterion, loop, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses1 = AverageMeter('Loss poisoned', ':.4e')
    losses2 = AverageMeter('Loss imagenet', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    imagenet_top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, losses1, losses2, top1, imagenet_top1],
        prefix="Epoch: [{}]".format(epoch))

    ### switch to train mode ###
    prompter.eval()
    model.train()

    num_batches_per_epoch = len(train_loader)

    start = time.time()
    end = start
    for i, ((imagenet_images, imagenet_target), (images, target)) in enumerate(tqdm(zip(imagenet_train_loader, train_loader))):

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        imagenet_images = imagenet_images.to(device)
        imagenet_target = imagenet_target.to(device)
        images = images.to(device)
        target = target.to(device)

        imagenet_output = model(imagenet_images)

        prompted_images = prompter(images)
        output = model(prompted_images)
        if indices:
            output = output[:,indices]
        loss1 = criterion(output, target)
        loss2 = criterion(imagenet_output, imagenet_target)
        loss = args.theta * loss1 + args.alpha * loss2 

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        top1.update(acc1[0].item(), images.size(0))

        losses1.update(loss1.item(), images.size(0))
        losses2.update(loss2.item(), imagenet_images.size(0))
        losses.update(loss.item(), images.size(0)+imagenet_images.size(0))

        imagenet_acc1 = accuracy(imagenet_output, imagenet_target, topk=(1,))
        imagenet_top1.update(imagenet_acc1[0].item(), imagenet_images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, top1.avg

def prompt_learning(args, indices, train_loader, model, prompter, optimizer, scheduler, criterion, loop, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    ### switch to train mode ###
    prompter.train()
    model.eval()

    num_batches_per_epoch = len(train_loader)

    start = time.time()
    end = start
    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images = images.to(device)
        target = target.to(device)

        prompted_images = prompter(images)
        output = model(prompted_images)
        if indices:
            output = output[:,indices]
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, top1.avg


def validate(args, indices, val_loader, model, prompter, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original Acc@1', ':6.2f')
    top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1_org, top1_prompt],
        prefix='Validate: ')

    # switch to evaluation mode
    prompter.eval()
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):

            images = images.to(device)
            target = target.to(device)
            prompted_images = prompter(images)

            # compute output
            output_prompt = model(prompted_images)
            output_org = model(images)
            if indices:
                output_prompt = output_prompt[:, indices]
                output_org = output_org[:, indices]
            loss = criterion(output_prompt, target)

            # measure accuracy and record loss
            acc1_org = accuracy(output_org, target, topk=(1,))
            acc1_prompt = accuracy(output_prompt, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1_org.update(acc1_org[0].item(), images.size(0))
            top1_prompt.update(acc1_prompt[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        progress.display(i)
        logging.info(' * Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
            .format(top1_prompt=top1_prompt, top1_org=top1_org))

    return top1_prompt.avg

def validate_imagenet(args, val_loader, model):
    batch_time = AverageMeter('Time', ':6.3f')
    top1_org = AverageMeter('Original Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1_org],
        prefix='Validate: ')

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):

            images = images.to(device)
            target = target.to(device)

            # compute output
            output_org = model(images)

            # measure accuracy and record loss
            acc1_org = accuracy(output_org, target, topk=(1,))
            top1_org.update(acc1_org[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        progress.display(i)
        logging.info(' * Imagenet Acc@1 {top1_org.avg:.3f}'
            .format(top1_org=top1_org))

    return top1_org.avg


if __name__ == '__main__':
    main()