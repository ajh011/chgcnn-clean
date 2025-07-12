from matbench.bench import MatbenchBenchmark
import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample
from datetime import datetime
import random

import wandb
import sklearn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Subset, DataLoader

from cgcnn.load_data_retry import CIFData
from cgcnn.load_data_retry import collate_pool, get_train_val_test_loader, get_k_folds, get_nested_folds
from cgcnn.model_main import CrystalHypergraphConvNet

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[150], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')

parser.add_argument('--gamma', default=0.1, type=float,
                    metavar='gamma', help='gamma for multi-step scheduler')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--train-ratio', default=0.9, type=float, metavar='N',
                    help='percentage of train data to be loaded (default 0.9)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

parser.add_argument('--n-hconv', default=1, type=int, metavar='N',
                   help='number of hconv layers')
parser.add_argument('--bonds', help='include/exclude bond convolution', action='store_true')
parser.add_argument('--motifs', help='include/exclude motif convolution', action='store_true')
parser.add_argument('--triplets', help='include/exclude triplet convolution', action='store_true')
parser.add_argument('--nominal-target-name', default='y', type=str, metavar='nom-target-name',
        help='name for target for human reference')
parser.add_argument('--device-number', type=int, default=0,
                    help='Specify device number for cuda')
parser.add_argument('--folds', type=int, default=1,
        help='Num folds, def:1')
parser.add_argument('--num-workers', type=int, default=1,
        help='Num workers, def:1')
parser.add_argument('--seed', type=int, default=7,
        help='Random seed, def:7')
args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

def train_val_test_data_from_indexes(dataset, train_indexes, test_indexes, train_ratio=0.9, seed=7):
    n_train = round(len(train_indexes)*train_ratio)
    random.seed(seed)
    random.shuffle(train_indexes)
    val_indexes = train_indexes[n_train:]
    tr_indexes = train_indexes[:n_train]

    d_train = Subset(dataset, tr_indexes)
    d_val = Subset(dataset, val_indexes)
    d_test = Subset(dataset, test_indexes)

    return d_train, d_val, d_test

def main(matbench_task, fold):
    global args, best_mae_error
    data_dir = f'cgcnn/proc_dataset_{matbench_task.dataset_name}'
    # load data
    dataset = CIFData(data_dir)
    train_index, targets = matbench_task.get_train_and_val_data(fold)
    train_indexes = list(train_index.index)
    test_indexes  = list(matbench_task.get_test_data(fold, include_target=False).index)
    train_data, val_data, test_data =  train_val_test_data_from_indexes(dataset, train_indexes, test_indexes, args.train_ratio) 
    collate_fn = collate_pool
    #### Load data into DataLoaders/Batches
    train_loader = DataLoader(                                          train_data,
        collate_fn = collate_fn,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False, #args.pin_memory,
        generator=torch.Generator().manual_seed(args.seed)
    )

    val_loader = DataLoader(
        val_data,
        collate_fn = collate_fn,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_data,
        collate_fn = collate_fn,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )
    # obtain target value normalizer
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_target = [dataset[i][1] for i in range(len(dataset))]
        else:
            sample_target = [dataset[i][1] for i in
                                sample(range(len(dataset)), 500)]
        normalizer = Normalizer(torch.tensor(sample_target))

    # build model
    ex_data = dataset[0]
    orig_atom_fea_len = ex_data[0][0].shape[-1]
    nbr_fea_len = ex_data[0][1].shape[-1]
    print(f'args.bonds is {args.bonds}')
    print(f'args.motifs is {args.motifs}')
    print(f'args.triplets is {args.triplets}')
    model = CrystalHypergraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args.atom_fea_len,
                                n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len,
                                n_h=args.n_h,
                                n_hconv = args.n_hconv,
                                bonds = args.bonds,
                                motifs = args.motifs,
                                triplets = args.triplets,
                                classification=True if args.task ==
                                                       'classification' else False)
    if args.cuda:
        device = torch.device(f'cuda:{args.device_number}')
        torch.cuda.set_device(device)
        model.cuda()
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.folds == 1:
        # define loss func and optimizer
        if args.task == 'classification':
            criterion = nn.NLLLoss()
        else:
            criterion = nn.MSELoss()
        if args.optim == 'SGD':
            optimizer = optim.SGD(model.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.optim == 'Adam':
            optimizer = optim.Adam(model.parameters(), args.lr,
                                   weight_decay=args.weight_decay)
        else:
            raise NameError('Only SGD or Adam is allowed as --optim')
    
    
        scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                                gamma=args.gamma)

        best_filename = os.path.join('models', f'{datetime.now().strftime("%Y-%m-%dT%H:%M:%S.f%z")}{matbench_task.dataset_name}bs{args.batch_size}_best_{fold}')
        if args.bonds == True:
            best_filename += 'b'
        if args.triplets == True:
            best_filename += 't'
        if args.motifs == True:
            best_filename += 'm'
        best_filename += f'{args.n_conv}_model.pth.tar' 

        checkpoint_filename = os.path.join('models', f'{datetime.now().strftime("%Y-%m-%dT%H:%M:%S.f%z")}{matbench_task.dataset_name}bs{args.batch_size}_{fold}_')
        if args.bonds == True:
            checkpoint_filename += 'b'
        if args.triplets == True:
            checkpoint_filename += 't'
        if args.motifs == True:
            checkpoint_filename += 'm'
        checkpoint_filename += f'{args.n_conv}_checkpoint.pth.tar' 
        print(f'Resume training with {checkpoint_filename} in case of failure')


        for epoch in range(args.start_epoch, args.epochs):
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, normalizer, scheduler, fold = fold)
    
            # evaluate on validation set
            mae_error = validate(val_loader, model, criterion, normalizer, epoch, fold=fold)
    
            if mae_error != mae_error:
                print('Exit due to NaN')
                sys.exit(1)
    
            scheduler.step()
    
            # remember the best mae_eror and save checkpoint
            if args.task == 'regression':
                is_best = mae_error < best_mae_error
                best_mae_error = min(mae_error, best_mae_error)
            else:
                is_best = mae_error > best_mae_error
                best_mae_error = max(mae_error, best_mae_error)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mae_error': best_mae_error,
                'optimizer': optimizer.state_dict(),
                'normalizer': normalizer.state_dict(),
                'args': vars(args)
            }, is_best, filename = checkpoint_filename, best_filename = best_filename)
    
        # test best model
        print('---------Evaluate Model on Test Set---------------')
        torch.cuda.empty_cache()
        best_checkpoint = torch.load(best_filename)
        model.load_state_dict(best_checkpoint['state_dict'])

        test_outputs = validate(test_loader, model, criterion, normalizer, args.epochs, test=True, return_outputs = True)
        return test_outputs
    
def train(train_loader, model, criterion, optimizer, epoch, normalizer, scheduler, fold = None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         input[3].cuda(non_blocking=True),
                         input[4].cuda(non_blocking=True),
                         input[5].cuda(non_blocking=True),
                         input[6].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[7]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3],
                         input[4],
                         input[5],
                         input[6],
                         input[7])
        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = torch.squeeze(model(*input_var))
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target.cpu())
            losses.update(loss.data.cpu(), target.cpu().size(0))
            mae_errors.update(mae_error, target.cpu().size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
 
        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                )
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, accu=accuracies,
                    prec=precisions, recall=recalls, f1=fscores,
                    auc=auc_scores)
                )


    if args.task == 'regression':
        wandb.log({'train-mse-avg': losses.avg, 'train-mae-avg': mae_errors.avg, 'epoch': epoch, 'batch-time': batch_time.avg, 'lr': scheduler.get_last_lr(), 'fold':fold}) 
        mae_error = mae(normalizer.denorm(output.data.cpu()), target.cpu())
    else:
        wandb.log({'train-acc-avg':accuracies.avg, 'train-loss-avg': losses.avg, 'epoch': epoch, 'batch-time': batch_time.avg, 'lr': scheduler.get_last_lr()[0], 'train-recall-avg': recalls.avg, 'train-fscore-avg': fscores.avg, 'train-auc-scores': auc_scores.avg, 'fold':fold}) 
           
 
def validate(val_loader, model, criterion, normalizer, epoch, test=False, return_outputs=False, fold = None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    if return_outputs:
        outputs = []
    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                if args.cuda:
                    input_var = (Variable(input[0].cuda(non_blocking=True)),
                                 Variable(input[1].cuda(non_blocking=True)),
                                 input[2].cuda(non_blocking=True),
                                 input[3].cuda(non_blocking=True),
                                 input[4].cuda(non_blocking=True),
                                 input[5].cuda(non_blocking=True),
                                 input[6].cuda(non_blocking=True),
                                 [crys_idx.cuda(non_blocking=True) for crys_idx in input[7]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3],
                             input[4],
                             input[5],
                             input[6],
                             input[7])
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output = torch.squeeze(model(*input_var))
        if return_outputs:
            outputs.append(output.clone().detach())
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target.cpu())
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))

    if test:
        if args.task == 'regression':
            wandb.log({'test-mse-avg': losses.avg, 'test-mae-avg': mae_errors.avg, 'epoch': epoch, 'batch-time': batch_time.avg, 'fold':fold}) 
        else:
            wandb.log({'test-acc-avg':accuracies.avg, 'test-loss-avg': losses.avg, 'epoch': epoch, 'batch-time': batch_time.avg, 'test-recall-avg': recalls.avg, 'test-fscore-avg': fscores.avg, 'test-auc-scores': auc_scores.avg, 'fold':fold}) 
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
        if args.task == 'regression':
            wandb.log({'val-mse-avg': losses.avg, 'val-mae-avg': mae_errors.avg, 'epoch': epoch, 'batch-time': batch_time.avg, 'fold':fold}) 
        else:
            wandb.log({'val-acc-avg':accuracies.avg, 'val-loss-avg': losses.avg, 'epoch': epoch, 'batch-time': batch_time.avg, 'val-recall-avg': recalls.avg, 'val-fscore-avg': fscores.avg, 'val-auc-scores': auc_scores.avg, 'fold':fold}) 

    if return_outputs:
        outputs = torch.cat(outputs)
        if args.task == 'regression':
            print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                            mae_errors=mae_errors))
        else:
            print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                     auc=auc_scores))
        return list(outputs.cpu().detach().numpy())
    else:
        if args.task == 'regression':
            print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                            mae_errors=mae_errors))
            return mae_errors.avg
        else:
            print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                     auc=auc_scores))
            return auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    
    mb = MatbenchBenchmark(autoload=False, subset= [#'matbench_dielectric',
#                                                    'matbench_log_gvrh',
#                                                    'matbench_log_kvrh',
#                                                    'matbench_perovskites',
                                                    #'matbench_phonons',
                                                    #'matbench_jdft2d',
#                                                    'matbench_mp_e_form',
                                                    'matbench_mp_gap',
#                                                    'matbench_mp_is_metal'
])

    task_names = ''
    for task in mb.tasks:
        task.load()
        wandb.init(
            # set the wandb project where this run will be logged
            project="chgcnn-dirty-matbench",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.lr,
            "architecture": "Agg-dirty" ,
            "dataset": task.dataset_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size
            }
        )

        for fold in task.folds:

            if args.task == 'regression':
                best_mae_error = 1e10
            else:
                best_mae_error = 0.

            output = main(task, fold)
            task.record(fold, output)

        wandb.finish()
        task_names += str(task.dataset_name)+'_' 
        mb.to_file(f"my_models_benchmark_{task.dataset_name}T{datetime.now()}.json.gz")

    mb.to_file(f"my_models_benchmark_{task_names}T{datetime.now()}.json.gz")
