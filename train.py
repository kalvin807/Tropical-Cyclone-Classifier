import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

import h5py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

dataset_dir = '.'
sys.path.append(dataset_dir)
IMAGES_H5 = f'{dataset_dir}/images.hdf5'

LABELS = pd.read_csv(f"{dataset_dir}/labels_with_images.csv")
LABELS['year'] = pd.to_datetime(
    LABELS['datetime'], format='%Y-%m-%d %X').dt.year


def getImageFromH5(h5, row):
    img_name = f"{row['sequence']}{row['raw_index']}"
    return h5[img_name][()]


def prepare_dataset(labels, ratio, year):
    processed_labels = labels[['class', 'sequence', 'raw_index', 'year']]
    test_set = processed_labels[processed_labels['year'] >= year]
    traindev_set = processed_labels[processed_labels['year'] < year]
    train_set, dev_set = train_test_split(
        traindev_set, train_size=ratio)
    return {'train': train_set, 'dev': dev_set, 'test': test_set}


class TyDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.df)

    def getImg(self, idx):
        with h5py.File(f"{dataset_dir}/images.hdf5", 'r') as h5:
            return getImageFromH5(h5, self.df.iloc[idx])

    def __getitem__(self, idx):
        img = Image.fromarray(self.getImg(idx))
        label = self.df.iloc[idx]['class']
        image = self.transform(img)
        return image, label


class PretrainNet(nn.Module):
    def __init__(self):
        super(PretrainNet, self).__init__()
        # Define layers
        # Input shape: (Batch size, 1, 512, 512)
        # Convolution layer(s)
        self.conv = self.densenet()
        # FC layer(s)
        self.fc = nn.Sequential(
            nn.LeakyReLU(),
            # 1000 is the default output size in densenet
            nn.Linear(1000, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 6)
        )

    def densenet(self):
        model = models.densenet161(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        return model

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


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
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, criterion, optimizer, epoch, freq, gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top3, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc3 = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top3.update(acc3[0], input.size(0))

        # compute gradient and do Adam
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % freq == 0:
            progress.print(i)


def validate(val_loader, model, criterion, freq, gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top3,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc3 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top3.update(acc3[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % freq == 0:
                progress.print(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f}'
              .format(top1=top1, top3=top3))

    return top1.avg


def main_worker(gpu, args):
    global best_acc1

    best_acc1 = 0
    ddl = args['ddl']
    lr = args['lr']
    freq = args['freq']

    if ddl:
        dist.init_process_group(
            backend='nccl',
            world_size=args['world_size'],
            rank=gpu
        )

    # define model
    torch.cuda.set_device(gpu)
    model = args['model']
    model.cuda(gpu)

    if ddl:
        args['batch_size'] = int(args['batch_size']/args['gpus'])
        args['worker_size'] = int(
            (args['worker_size'] + args['gpus'] - 1) / args['gpus'])
        model = DDP(model, device_ids=[gpu])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    torch.backends.cudnn.benchmark = True

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        args['train_set']) if ddl else None

    train_loader = DataLoader(args['train_set'],
                              batch_size=args['batch_size'],
                              num_workers=args['worker_size'],
                              pin_memory=args['pin_memory'],
                              drop_last=True,
                              shuffle=(ddl is False),
                              train_sampler=train_sampler)

    dev_loader = DataLoader(args['train_set'],
                            batch_size=args['batch_size'],
                            num_workers=args['worker_size'],
                            pin_memory=args['pin_memory'],
                            drop_last=True,
                            shuffle=False)

    for epoch in range(0, args['epochs']):
        if ddl:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, freq, gpu)

        # evaluate on validation set
        acc1 = validate(dev_loader, model, criterion, freq, gpu)

        # remember best acc@1 and save checkpoint
        best_acc1 = max(acc1, best_acc1)


def main(args):
    args = {
        'model': PretrainNet(),
        'epochs': 50,
        'lr': 0.001,
        'freq': 100,
        'batch_size': 512,
        'worker_size': 8,
        'pin_memory': True,
        'ddl': False
    }

    dataset = prepare_dataset(LABELS, 0.2, 2012)

    args['train_set'] = dataset['train']
    args['dev_set'] = dataset['dev']

    if torch.cuda.device_count() > 1:
        args['gpus'] = torch.cuda.device_count()
        print("Distribute work to {} GPU".format(args['gpus']))
        # Planning to use 1 node + multi-GPU only, so world size = GPU count * 1
        args['world_size'] = args['gpus']
        args['ddl'] = True
        # Set up Master node listening port/addr
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8889'
        # Spawn processes according to world_size
        mp.spawn(main_worker, nprocs=args['gpus'], args=args)
    else:
        main_worker(0, args)


if __name__ == "__main__":
    main()
