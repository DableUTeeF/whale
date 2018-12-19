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
from sklearn.metrics import f1_score


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


if __name__ == '__main__':
    colors = ['\033[0m',
              '\033[31m', '\033[32m', '\033[33m', '\033[34m', '\033[35m',
              '\033[36m', '\033[37m', '\033[91m', '\033[94m', '\033[95m']
    args = DotDict({
        'batch_size': 32,
        'batch_mul': 1,
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
    train_list = lists[:int(lslength*0.8)]
    test_list = lists[int(lslength*0.8):]

    try:
        os.listdir('/root')
        rootpath = '/root/palm/DATA/HPAIC/train'
    except PermissionError:
        rootpath = '/media/palm/data/Human Protein Atlas/train'
    train_dataset = datagen.Generator(train_list,
                                      rootpath,
                                      (512, 224),
                                      normalize=normalize,
                                      )
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=False)
    test_dataset = datagen.Generator(test_list,
                                     rootpath,
                                     (512, 224),
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
        print(color+'Epoch: %d/%d' % (epoch, args.epochs))
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()
        start_time = time.time()
        last_time = start_time
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            # inputs = normalize(inputs)
            outputs = model(inputs)
            # targets = torch.cat((targets, targets, targets, targets, targets))

            # bs, ncrops, c, h, w = inputs.size()
            # outputs = outputs.view(bs, ncrops, -1).mean(1)

            loss = bcecriterion(outputs[0], targets.float()) / args.batch_mul / 2
            loss += ccecriterion(outputs[1], targets.float()) / args.batch_mul / 2
            loss.backward()
            train_loss += loss.item() * args.batch_mul
            _, predicted = outputs.max(1)
            total += targets.size(0)
            ytrue = targets.cpu().detach().numpy().flatten()
            ypred = outputs.cpu().detach().numpy().flatten().round()
            correct += f1_score(ytrue, ypred, average='macro')
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
                  f'loss: {train_loss / (batch_idx + 1):.{3}} - ' + \
                  f'acc: {correct / (batch_idx+1):.{5}}'
            print(f'\r{color}{lss}', end=colors[0])

        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
            logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)
        logger.scalar_summary('acc', correct / total, epoch)
        logger.scalar_summary('loss', train_loss / (batch_idx + 1), epoch)
        color = colors[np.random.randint(1, len(colors))]
        print(f'\r {color}'
              f'ToT: {format_time(time.time() - start_time)} - '
              f'loss: {train_loss / (batch_idx + 1):.{3}} - '
              f'acc: {correct / (batch_idx + 1):.{5}}', end='')
        optimizer.step()
        optimizer.zero_grad()
        # scheduler2.step()


    def test(epoch):
        global best_acc, best_no
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                # inputs = normalize(inputs)
                outputs = model(inputs)
                loss = bcecriterion(outputs[0], targets.float()) / 2
                loss += ccecriterion(outputs[1], targets.float()) / 2
                """"""
                # bs, ncrops, c, h, w = inputs.size()
                # outputs = outputs.view(bs, ncrops, -1).mean(1)
                """"""
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                ytrue = targets.cpu().detach().numpy().flatten()
                ypred = outputs.cpu().detach().numpy().flatten().round()
                correct += f1_score(ytrue, ypred, average='macro')

                # progress_bar(batch_idx, len(val_loader), 'Acc: %.3f%%'
                #              % (100. * correct / total))
        logger.scalar_summary('val_acc', correct / total, epoch)
        logger.scalar_summary('val_loss', test_loss / (batch_idx + 1), epoch)
        print(f'{c} - val_acc: {correct / total:.{5}}{colors[0]}')
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
