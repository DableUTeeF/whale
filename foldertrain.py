from __future__ import print_function
import os
import warnings

warnings.simplefilter("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from utils import Logger, format_time
import models
from torch.optim.lr_scheduler import MultiStepLR
import time
import torchvision.datasets as datasets
from sys import stdout


def getmodel(cls=61):
    model_conv = models.densenet.densenet121(pretrained=True)
    num_ftrs = model_conv.classifier.in_features
    model_conv.classifier = nn.Linear(num_ftrs, cls)
    return model_conv


class DotDict(dict):
    def __getattr__(self, name):
        return self[name]


def normalize(x):
    x = x.astype('float32') / 255
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    for i in range(3):
        x[i, :, :] -= mean[i]
        x[i, :, :] /= std[i]
    return x


def printstd(args, end=None):
    stdout.write(args)
    if end:
        stdout.write(end)
    else:
        stdout.write('\n')
    stdout.flush()


if __name__ == '__main__':
    colors = ['\033[0m',
              '\033[31m', '\033[32m', '\033[33m', '\033[34m', '\033[35m',
              '\033[36m', '\033[37m', '\033[91m', '\033[94m', '\033[95m']
    args = DotDict({
        'batch_size': 16,
        'batch_mul': 2,
        'val_batch_size': 16,
        'cuda': True,
        'model': '',
        'train_plot': False,
        'epochs': 90,
        'try_no': 'densenet121_folder',
        'imsize': 224,
        'imsize_l': 256,
        'traindir': '/root/palm/DATA/plant/train',
        'valdir': '/root/palm/DATA/plant/validate',
        'workers': 16,
        'resume': False,
    })
    try:
        length = os.listdir(f'./logs/')
        le = 0
        for f in length:
            if f.startswith(args.try_no):
                le += 1
    except FileNotFoundError:
        le = 0
    logger = Logger(f'./logs/{args.try_no}_{le}')
    logger.text_summary('Describe', 'DenseNet201', 0)
    logger.text_summary('Describe', 'Batch size: 32*1', 1)
    logger.text_summary('Describe', 'Input size: 224,224', 2)
    logger.text_summary('Describe', 'From folder', 3)
    best_acc = 0
    best_no = 0
    start_epoch = 1
    model = getmodel(5004).cuda()

    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4,
                                # nesterov=False,
                                )
    scheduler = MultiStepLR(optimizer, [30, 60, 90, 120, 150])
    criterion = nn.CrossEntropyLoss().cuda()
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    try:
        os.listdir('/root')
        rootpath = '/root/palm/DATA/whale/foldered/train'
    except PermissionError:
        rootpath = '/media/palm/data/whale/foldered/train'

    train_dataset = datasets.ImageFolder(
        rootpath,
        transforms.Compose([
            transforms.Resize(256),
            # ReplicatePad(args.imsize_l[i]),
            transforms.CenterCrop(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=False)

    # model = torch.nn.parallel.DistributedDataParallel(model).cuda()
    # model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = args.batch_size > 1
    if args.resume:
        if args.resume is True:
            args['resume'] = f'./checkpoint/try_{args.try_no}best.t7'
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    c = 0


    def train(epoch):
        color = colors[np.random.randint(1, len(colors))]
        print(color + 'Epoch: %d/%d' % (epoch, args.epochs))
        model.train()
        cce_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()
        start_time = time.time()
        last_time = start_time
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model(inputs)

            loss = criterion(outputs, targets.long()) / args.batch_mul
            cce_loss += loss.item() * args.batch_mul
            loss.backward()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += sum(predicted.eq(targets).cpu().detach().numpy().flatten())
            lfs = (batch_idx + 1) % args.batch_mul
            if lfs == 0:
                optimizer.step()
                optimizer.zero_grad()
            step_time = time.time() - last_time
            last_time = time.time()
            try:
                print(f'\r{" " * (len(lss))}', end='')
            except NameError:
                pass
            color = colors[np.random.randint(1, len(colors))]
            lss = f'{batch_idx}/{len(trainloader)} | ' + \
                  f'ETA: {format_time(step_time * (len(trainloader) - batch_idx))} - ' + \
                  f'cce_loss: {cce_loss / (batch_idx + 1):.{3}} - ' + \
                  f'acc: {correct / total:.{5}}'
            print(f'\r{color}{lss}', end=colors[0])

        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
            logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)
        logger.scalar_summary('acc', correct / total, epoch)
        logger.scalar_summary('cce_loss', cce_loss / (batch_idx + 1), epoch)
        color = colors[np.random.randint(1, len(colors))]
        print(f'\r{" " * (len(lss))}', end='')
        print(f'\r {color}'
              f'ToT: {format_time(time.time() - start_time)} - '
              f'cce_loss: {cce_loss / (batch_idx + 1):.{3}} - '
              f'acc: {correct / total:.{5}}{colors[0]}')
        optimizer.step()
        optimizer.zero_grad()

        # print('Saving..')
        state = {
            'optimizer': optimizer.state_dict(),
            'net': model.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/try_{args.try_no}temp.t7')


    for epoch in range(start_epoch, start_epoch + args.epochs):
        scheduler.step()
        train(epoch)
        color = colors[np.random.randint(1, len(colors))]
        # print(f'{color}best: {best_acc}{colors[0]}')
    start_epoch += args.epochs
