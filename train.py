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
import datagen
from sys import stdout


def getmodel(cls=61):
    model_conv = models.densenet.densenet201(pretrained=True)
    model_conv.features.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
        'batch_size': 1,
        'batch_mul': 24,
        'val_batch_size': 1,
        'cuda': True,
        'model': '',
        'train_plot': False,
        'epochs': 90,
        'try_no': 'densenet201_1',
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
    logger.text_summary('Describe', 'Input size: 224/256', 2)
    logger.text_summary('Describe', 'sigmoid+softmax', 3)
    best_acc = 0
    best_no = 0
    start_epoch = 1
    model = models.whale.WhaleModelA().cuda()

    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4,
                                # nesterov=False,
                                )
    scheduler = MultiStepLR(optimizer, [30, 60, 90, 120, 150])
    bcecriterion = nn.BCEWithLogitsLoss().cuda()
    ccecriterion = nn.CrossEntropyLoss().cuda()
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # manage list
    csv = open('train.csv', 'r').readlines()
    lists = []
    for line in csv[1:]:
        id_, target = line[:-1].split(',')
        lists.append((id_, target.split()))
    lslength = len(lists)
    ratio = 0.8
    train_list = lists[:int(lslength * 0.8)]
    test_list = lists[int(lslength * 0.8):]

    try:
        os.listdir('/root')
        rootpath = '/root/palm/DATA/whale/train'
    except PermissionError:
        rootpath = '/media/palm/data/whale/train'
    train_dataset = datagen.Generator(train_list,
                                      rootpath,
                                      (224, 224),
                                      normalize=normalize,
                                      )
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=False)
    test_dataset = datagen.Generator(test_list,
                                     rootpath,
                                     (224, 224),
                                     normalize=normalize,
                                     )
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.val_batch_size,
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
            best_prec1 = checkpoint['acc']
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
        bce_loss = 0
        cce_loss = 0
        correct = 0
        total = 0
        diciding = 0
        optimizer.zero_grad()
        start_time = time.time()
        last_time = start_time
        for batch_idx, (inputs, dicider, targets) in enumerate(trainloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            dicider = dicider.to('cuda')
            outputs = model(inputs)

            if dicider.cpu().detach().numpy()[0] == 0:
                loss = bcecriterion(outputs[0], dicider.float()) / args.batch_mul
                bce_loss += loss.item() * args.batch_mul
            else:
                loss = ccecriterion(outputs[1], targets.long()) / args.batch_mul
                cce_loss += loss.item() * args.batch_mul
            loss.backward()
            _, predicted = outputs[1].max(1)
            diciding += sum(outputs[0].round().eq(dicider.float()).cpu().detach().numpy().flatten())
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
                  f'bce_loss: {bce_loss / (batch_idx + 1):.{3}} - ' + \
                  f'cce_loss: {cce_loss / (batch_idx + 1):.{3}} - ' + \
                  f'dicider: {diciding / total:.{3}} - ' + \
                  f'acc: {correct / total:.{5}}'
            print(f'\r{color}{lss}', end=colors[0])

        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
            logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)
        logger.scalar_summary('acc', correct / total, epoch)
        logger.scalar_summary('bce_loss', bce_loss / (batch_idx + 1), epoch)
        logger.scalar_summary('cce_loss', cce_loss / (batch_idx + 1), epoch)
        color = colors[np.random.randint(1, len(colors))]
        print(f'\r {color}'
              f'ToT: {format_time(time.time() - start_time)} - '
              f'bce_loss: {bce_loss / (batch_idx + 1):.{3}} - '
              f'cce_loss: {cce_loss / (batch_idx + 1):.{3}} - '
              f'dicider: {diciding / total:.{3}} - '
              f'acc: {correct / total:.{5}}', end='')
        optimizer.step()
        optimizer.zero_grad()
        # scheduler2.step()


    def test(epoch):
        global best_acc, best_no
        model.eval()
        bce_loss = 0
        cce_loss = 0
        correct = 0
        total = 0
        diciding = 0
        with torch.no_grad():
            for batch_idx, (inputs, dicider, targets) in enumerate(val_loader):
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                dicider = dicider.to('cuda')
                # inputs = normalize(inputs)
                outputs = model(inputs)
                if dicider.cpu().detach().numpy()[0] == 0:
                    loss = bcecriterion(outputs[0], dicider.float())
                    bce_loss += loss.item()
                else:
                    loss = ccecriterion(outputs[1], targets.long())
                    cce_loss += loss.item()
                _, predicted = outputs[1].max(1)
                diciding += sum(outputs[0].round().eq(dicider.float()).cpu().detach().numpy().flatten())
                total += targets.size(0)
                correct += sum(predicted.eq(targets).cpu().detach().numpy().flatten())

                # progress_bar(batch_idx, len(val_loader), 'Acc: %.3f%%'
                #              % (100. * correct / total))
        logger.scalar_summary('val_bce_loss', bce_loss / (batch_idx + 1), epoch)
        logger.scalar_summary('val_cce_loss', cce_loss / (batch_idx + 1), epoch)
        print(f'{c} - val_acc: {correct / total:.{5}}- val_dic: {diciding / total:.{5}}{colors[0]}')
        # platue.step(correct)

        # Save checkpoint.
        acc = 100. * correct / total
        # print('Saving..')
        state = {
            'optimizer': optimizer.state_dict(),
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if acc > best_acc:
            torch.save(state, f'./checkpoint/try_{args.try_no}best.t7')
            best_acc = acc
            best_no = correct
        torch.save(state, f'./checkpoint/try_{args.try_no}temp.t7')


    for epoch in range(start_epoch, start_epoch + args.epochs):
        scheduler.step()
        train(epoch)
        test(epoch)
        color = colors[np.random.randint(1, len(colors))]
        print(f'{color}best: {best_acc}{colors[0]}')
    start_epoch += args.epochs
