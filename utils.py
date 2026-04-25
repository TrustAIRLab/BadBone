import shutil
import os
import torch
import numpy as np
import argparse
import logging

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for Vision Models')

    parser.add_argument('--root_path', type=str, default=ROOT_PATH)

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency in an epoch')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=40,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.1,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=10)

    # model
    parser.add_argument('--model', type=str, default=None,
                        choices=['rn50', 'instagram_resnext101_32x8d', 'bit_m_rn50', 'rn18'],
                        help='choose pre-trained model')
    parser.add_argument('--method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch'],
                        help='choose visual prompting method')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')

    # dataset
    parser.add_argument('--root', type=str, default=f'{ROOT_PATH}/data',
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset: cifar100, cifar10, svhn, eurosat, stl10')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')
    parser.add_argument('--num_class', type=int, default=10,
                        help='number of classes')

    # other
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default=f'{ROOT_PATH}/save/prompters',
                        help='path to save prompters')
    parser.add_argument('--image_dir', type=str, default=f'{ROOT_PATH}/save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=str, default="1",
                        help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')
    parser.add_argument('--use_wandb', default=False,
                        action="store_true",
                        help='whether to use wandb')

    ### backdoor data generation
    parser.add_argument('--target_label', type=int, default=1,
                        help='target label of backdoor attack')
    parser.add_argument('--poison_portion', type=float, default=5000,
                        help='poison rate of original dataset')
    parser.add_argument('--clean_portion', type=float, default=5000,
                        help='rate of original dataset contained')
    parser.add_argument('--patch_size', type=int, default=10,
                        help='size of trigger')
    parser.add_argument('--patch_mode', type=str, default='fix',
                        choices=['fix', 'center'],
                        help='mode for creating backdooring triggers')
    parser.add_argument('--label_mode', type=str, default='target',
                        choices=['target', 'untarget_next', 'untarget_random'],
                        help='mode for creating backdoor labels')
    parser.add_argument('--backdoor_mode', type=str, default='mix',
                        choices=['bd_only', 'with_same_part', 'mix', 'untarget', 'untarget_loss', 'untarget_random', 'emb'],
                        help='mode for creating backdoored dataset')
    ### backdoor model finetuning
    parser.add_argument('--num_loops', type=int, default=2,
                        help='Number of loops of the backdoor process')
    parser.add_argument('--bd_epochs', type=int, default=10,
                        help='number of training epochs in finetuning')
    parser.add_argument('--bd_learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument("--bd_weight_decay", type=float, default=0.005,
                        help="weight decay")
    parser.add_argument("--bd_warmup", type=int, default=5,
                        help="number of steps to warmup for")
    parser.add_argument('--bd_momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--bd_patience', type=int, default=5)

    ### train with Imagenet 
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='weight for (loss imagenet) training imagenet')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='weight for (loss clean) training clean data in the untarget backdoor setting (loss based)')
    parser.add_argument('--theta', type=float, default=1.0,
                        help='weight for (loss triggered) training triggered data in the untarget backdoor setting (emb based)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='weight for (loss embedding) training triggered data embedding loss in the untarget backdoor setting (emb based)')

    ### save path
    parser.add_argument('--bd_data_dir', type=str, default=f'{ROOT_PATH}/save/data',
                        help='path to save backdoored data')
    parser.add_argument('--bd_dir', type=str, default=f'{ROOT_PATH}/save/bd',
                        help='path to save backdoored checkpoints')
    parser.add_argument('--pretrained_model_dir', type=str, default=f'{ROOT_PATH}/save/pretrained_models',
                        help='path to save pretrained models')
    parser.add_argument('--resume_pretrained_model', type=str, default=None,
                        help='path to the pretrained model')   

    ### how to train backdoor model
    parser.add_argument('--prompt_only', default=False, action="store_true",
                        help='only prompting') 
    parser.add_argument('--finetune_only', default=False, action="store_true",
                        help='only finetuning the model')          

    parser.add_argument('--label_map', type=str, default='top',
                        choices=['top', 'random', 'semantic', 'bottom'],
                        help='how to map labels in prompting')
    
    ### how to get close to embedding ###
    parser.add_argument('--emb_method', type=str, default='backbone',
                        choices=['backbone', 'prompt'],
                        help='how to embed target images')

    args = parser.parse_args()

    args.filename = '{}_{}_{}_{}_{}_lp{}_l{}_d{}_bl{}_bd{}_bsz{}_trial_{}'. \
        format(args.method, args.prompt_size, args.dataset, args.model,
               args.optim, args.num_loops,
               args.learning_rate, args.weight_decay, 
               args.bd_learning_rate, args.bd_weight_decay, args.batch_size, 
               args.trial)

    args.model_folder = os.path.join(args.model_dir, args.filename)
    # if not os.path.isdir(args.model_folder):
    #     os.makedirs(args.model_folder)
    args.pretrained_model_folder = os.path.join(args.pretrained_model_dir, args.filename)
    # if not os.path.isdir(args.pretrained_model_folder):
    #     os.makedirs(args.pretrained_model_folder)

    args.image_folder = os.path.join(args.image_dir, args.filename)
    # if not os.path.isdir(args.image_folder):
    #     os.makedirs(args.image_folder)

    args.data_filename = '{}_{}_{}_{}_{}_{}_{}'. \
        format(args.dataset, args.label_mode, args.patch_mode, args.patch_size, 
                args.target_label, args.poison_portion, args.clean_portion,
               )

    args.bd_data_folder = os.path.join(args.bd_data_dir, args.data_filename)

    return args


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names

def save_backdoor_checkpoint(loop, epoch, mode, state, args, is_best=False):
    if not os.path.isdir(args.bd_folder):
        os.makedirs(args.bd_folder)
    dirname = '{}_{}'.format(mode, loop)
    if not os.path.isdir(os.path.join(args.bd_folder, dirname)):
        os.makedirs(os.path.join(args.bd_folder, dirname))
    filename = 'checkpoint{}.pth.tar'.format(epoch) ### TODO Done
    savefile = os.path.join(args.bd_folder, dirname, filename)
    bestfile = os.path.join(args.bd_folder, dirname, 'model_best.pth.tar')
    torch.save(state, savefile)
    logging.info("Checkpoint ({}) of epoch {} in loop {} saved.".format(mode, epoch, loop))
    if is_best:
        shutil.copyfile(savefile, bestfile)
        logging.info('Best checkpoint ({}) saved.'.format(mode))


def load_backdoor_checkpoint(loop, epoch, mode, args, is_best=False):
    dirname = '{}_{}'.format(mode, loop)
    filename = 'checkpoint{}.pth.tar'.format(epoch) ### TODO Done
    savefile = os.path.join(args.bd_folder, dirname, filename)
    bestfile = os.path.join(args.bd_folder, dirname, 'model_best.pth.tar')
    if is_best:
        model = torch.load(bestfile)
    else:
        model = torch.load(savefile)
    return model

def save_checkpoint(state, args, is_best=False, filename='checkpoint.pth.tar'):
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)
    savefile = os.path.join(args.model_folder, filename)
    bestfile = os.path.join(args.model_folder, 'model_best.pth.tar')
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, bestfile)
        logging.info ('saved best file')

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
