import glob
import os
import shutil
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
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

dataset_dir = '.'
sys.path.append(dataset_dir)

# Global Variables
IMAGES_H5 = f'{dataset_dir}/images.hdf5'
NPY_FOLDER = f'{dataset_dir}/uint8'
LABELS = pd.read_csv(f"{dataset_dir}/labels_with_images.csv")
LABELS['year'] = pd.to_datetime(
    LABELS['datetime'], format='%Y-%m-%d %X').dt.year
LABELS = LABELS[LABELS['class'] < 5]

def getImageFromH5(h5, row):
    img_name = f"{row['sequence']}{row['raw_index']}"
    return h5[img_name][()]


def prepare_dataset(labels, ratio, year):
    processed_labels = labels[['class', 'sequence', 'raw_index', 'year']]
    test_set = processed_labels[processed_labels['year'] >= year]
    traindev_set = processed_labels[processed_labels['year'] < year]
    train_set, dev_set = train_test_split(
        traindev_set, train_size=ratio, shuffle=True)
    return {'train': train_set, 'dev': dev_set, 'test': test_set}


class TyDataset(Dataset):
    # Make sure you config the dataset properly before you train
    # TODO: Make it more general and configable
    def __init__(self, df, transform, channel=1):
        self.df = df
        self.transform = transform
        self.channel = channel
        self.sequence = df['sequence'].values
        self.raw_index = df['raw_index'].values

    def __len__(self):
        return len(self.df)

    def getImg(self, idx):
        # If you are using npy
        return np.load(f"{NPY_FOLDER}/{self.sequence[idx]}{self.raw_index[idx]}.npy")

    # When use against with DenseNet you need to make img_arr to have 3 channels, to do so
    # either by PIL convert('RGB') or use numpy stack(, axis=-1)
    def __getitem__(self, idx):
        img_arr = self.getImg(idx)
        if self.channel == 3:
            img = Image.fromarray(np.stack((img_arr,)*3, axis=-1))
        else:
            img = Image.fromarray(img_arr)
        image = self.transform(img)
        label = self.df.iloc[idx]['class']
        return image, label


class EffNet(nn.Module):
    # Of course you need to check you are training the correct model or not
    def __init__(self):
        super(EffNet, self).__init__()

        self.conv = EfficientNet.from_pretrained('efficientnet-b3')
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(1536, 1024),
            nn.Linear(1024,6)
        )

    def forward(self, x):
        x = self.conv.extract_features(x)
        x = self.fc(x)
        return x


class AverageMeter(object):  # Helper functions for training loops
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


def accuracy(output, target):
    """Computes the accuracy over the top predictions"""
    with torch.no_grad():
        batch_size = target.size(0)

        _, preds = torch.max(output.data, 1)
        correct = (preds == target).sum().item()

        return correct/batch_size


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch, freq, gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             prefix="Epoch: [{}]".format(epoch))

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
        acc1 = accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))

        # compute gradient and do Adam
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % freq == 0:
            progress.print(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, freq, gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time,
                             losses, top1, prefix='Test: ')

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
            acc1 = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(acc1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % freq == 0:
                progress.print(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg


def main_worker(gpu, args):
    global best_acc1

    print("Worker {} starts".format(gpu))

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
        model = DDP(model, device_ids=[gpu], find_unused_parameters=False)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    # optimizer = torch.optim.Adam(model.parameters(), lr)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr,
                                momentum=0.9,
                                weight_decay=0.0001,
                                nesterov = True)

    torch.backends.cudnn.benchmark = True

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        args['train_set']) if ddl else None

    train_loader = DataLoader(args['train_set'],
                              batch_size=args['batch_size'],
                              num_workers=args['worker_size'],
                              pin_memory=args['pin_memory'],
                              drop_last=True,
                              shuffle=(ddl is False),
                              sampler=train_sampler)

    dev_loader = DataLoader(args['dev_set'],
                            batch_size=args['batch_size'],
                            num_workers=args['worker_size'],
                            pin_memory=args['pin_memory'],
                            drop_last=True,
                            shuffle=False)

    train_losses = []
    train_accs = []
    dev_losses = []
    dev_accs = []

    for epoch in range(0, args['epochs']):
        if ddl:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch
        acc1, loss = train(train_loader, model, criterion,
                     optimizer, epoch, freq, gpu)
        train_accs.append(acc1)
        train_losses.append(loss)

        # evaluate on validation set
        acc1, loss = validate(dev_loader, model, criterion, freq, gpu)
        dev_accs.append(acc1)
        dev_losses.append(loss)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not ddl or (ddl and gpu == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

    # Save stat for analysis usage
    print('Train Loss over epoch :', train_losses)
    print('Train Acc over epoch :', train_accs)
    print('Dev Loss over epoch :', dev_losses)
    print('Dev Acc over epoch :', dev_accs)
    data = {'Train_Loss': train_losses, 'Train_Accuracy': train_accs,
	    'Dev_Loss': dev_losses, 'Dev_Accuracy': dev_accs}
    pd.DataFrame(data=data).to_csv(f'{dataset_dir}/train_stats-{gpu}.csv')

def main():
  # Define arguments here
    args = {
        'model': EffNet(),
        'epochs': 100,
        'lr': 0.1,
        'freq': 100,
        'batch_size': 56,
        'worker_size': 16,
        'pin_memory': True,
        'ddl': False
    }

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dev_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = prepare_dataset(LABELS, 0.8, 2012)
    args['train_set'] = TyDataset(dataset['train'], train_transform, channel=3)
    args['dev_set'] = TyDataset(dataset['dev'], dev_transform, channel=3)

    if torch.cuda.device_count() > 1:
        args['gpus'] = torch.cuda.device_count()
        print("Distribute work to {} GPU".format(args['gpus']))
        # Planning to use 1 node + multi-GPU only, so world size = GPU count * 1
        args['world_size'] = args['gpus']
        args['ddl'] = True
        # Set up Master node listening port/addr
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8899'
        # Spawn processes according to world_size
        mp.spawn(main_worker, nprocs=args['gpus'], args=(args,))
    else:
        main_worker(0, args)

    print("--------------------------")
    print("*** Testing you know ***")
    test_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
    test_ds = TyDataset(dataset['test'], test_transform, channel=3)
    test_loader = DataLoader(test_ds,
                            batch_size = 32,
                            num_workers= 16,
                            pin_memory= True,
                            drop_last=True,
                            shuffle=False)
    _, _ = validate(val_loader=test_loader, model=args['model'],
		    criterion=nn.CrossEntropyLoss().cuda(0), freq=100, gpu=0)

if __name__ == "__main__":
    main()
