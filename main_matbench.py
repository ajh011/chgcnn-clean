import time
import torch, shutil, argparse
import torch.optim as optim
import numpy as np
import wandb
from torch_geometric.nn import to_hetero
from torch_geometric.loader import DataLoader
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from data.matbench_dataset import InMemoryCrystalHypergraphDataset
from model.chgcnn import CrystalHypergraphConv
from data.hypergraph.hypergraph import *
import torch_geometric.transforms as T
from datetime import datetime

from torch.utils.data import Subset

from random import sample
import random

from matbench.bench import MatbenchBenchmark

#Define default dataset here for convenience (target name
#derived from name here, unless otherwise specified)
default_dataset = 'dataset_log_k'

def ClassificationAccuracy(output, target):
    prediction = torch.argmax(output, dim=1)
    acc = (prediction == target).float().mean()
    return acc


class AverageMeter:
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


class ProgressMeter:

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train(model, device, train_loader, loss_criterion, accuracy_criterion, optimizer, epoch, task, target_name, normalizer, scheduler):
    batch_time = AverageMeter('Batch', ':.4f')
    data_time = AverageMeter('Data', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    accus = AverageMeter('Accu', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, accus],
        prefix='Epoch: [{}]'.format(epoch))

    model.train()

    end = time.time()

    for i, data in enumerate(train_loader):

        data_time.update(time.time() - end)
        data = data.to(device, non_blocking=True)
        output = torch.squeeze(model(data))
        if task == 'regression':
            target = torch.tensor([float(i) for i in data.y]).to(device)
            target_norm = normalizer.norm(target)

            target = torch.squeeze(target)
            target_norm = torch.squeeze(target_norm)

            loss = loss_criterion(output, target_norm)
            accu = accuracy_criterion(normalizer.denorm(output), target)
            losses.update(loss.item(), target_norm.size(0))
            accus.update(accu.item(), target.size(0))
        else:
            target = data.y.long()
            loss = loss_criterion(output, target)
            accu = accuracy_criterion(output, target.float())
            losses.update(loss.item(), target.size(0))
            accus.update(accu.item(), target.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)
    if task == 'regression':
        wandb.log({'train-mse-avg': losses.avg, 'train-mae-avg': accus.avg, 'epoch': epoch, 'batch-time': batch_time.avg, 'lr': scheduler.get_last_lr()}) 
    else:
        wandb.log({'train-nll-avg': losses.avg, 'train-accuracy-avg': accus.avg, 'epoch': epoch, 'batch-time': batch_time.avg}) 
    return losses.avg, accus.avg

def validate(model, device, val_loader, loss_criterion, accuracy_criterion, epoch, task, target_name, normalizer, test = False, return_outs = False):
    batch_time = AverageMeter('Batch', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    accus = AverageMeter('Accu', ':.4f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, accus],
        prefix='Val: ')

    model.eval()
    if return_outs:
        outputs = []

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):

            data = data.to(device, non_blocking=True)
            output = torch.squeeze(model(data))
            if task == 'regression':
                target = torch.tensor([float(i) for i in data.y]).to(device)
                target_norm = normalizer.norm(target)
                target = torch.squeeze(target)
                target_norm = torch.squeeze(target_norm)

                loss = loss_criterion(output, target_norm)
                accu = accuracy_criterion(normalizer.denorm(output), target)
                losses.update(loss.item(), target_norm.size(0))
                accus.update(accu.item(), target.size(0))
            else:
                target = data.y.long()
                loss = loss_criterion(output, target)
                accu = accuracy_criterion(output, target)
                losses.update(loss.item(), target.size(0))
                accus.update(accu.item(), target.size(0))

            if return_outs:
                outputs.append(output)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)
    if test == False:
        if task == 'regression':
            wandb.log({'val-mse-avg': losses.avg, 'val-mae-avg': accus.avg, 'i': i, 'batch-time': batch_time.avg, 'epoch': epoch}) 
        else:
            wandb.log({'val-nll-avg': losses.avg, 'val-accu-avg': accus.avg, 'i': i, 'batch-time': batch_time.avg, 'epoch': epoch}) 
    elif test == True:
        if task == 'regression':
            wandb.log({'test-mse-avg': losses.avg, 'test-mae-avg': accus.avg, 'i': i, 'batch-time': batch_time.avg, 'epoch': epoch}) 
        else:
            wandb.log({'test-nll-avg': losses.avg, 'test-accu-avg': accus.avg, 'i': i, 'batch-time': batch_time.avg, 'epoch': epoch})
    print(f'Total Acc: {accus.avg}\nTotal Loss:{losses.avg}')
    if return_outs:
        outputs = torch.cat(outputs)
        return list(outputs.cpu())
    else:
        return accus.avg

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
    parser = argparse.ArgumentParser(description='CGCNN')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='trainning ratio')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--milestones', type=list, default=[150], metavar='Mlstn',
                        help='milestones for multistep scheduler (default: 150)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 5e-5)',
                        dest='weight_decay')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed (default: 7)')
    parser.add_argument('--print-freq', default=1, type=int,
                        metavar='N', help='print frequency (default: 1)')
    parser.add_argument('--optim', default='SGD', type=str, metavar='Adam',
                        help='choose an optimizer, SGD or Adam, (default:Adam)')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--task', default='regression', type=str)
    parser.add_argument('--num_class', default=2, type=int)
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--drop-last', default=False, type=bool)
    parser.add_argument('--pin-memory', default=False, type=bool)
    parser.add_argument('--dir', default=f'data/{default_dataset}', type=str)
    parser.add_argument('--normalize', default=True, type=bool)
    parser.add_argument('--target_name', default=str(default_dataset), type=str, help='formation energy (form_en), band gap (band_gap) or energy above hull (en_abv_hull) prediction task') 
    parser.add_argument('--scheduler', default=True, type=bool,
            help = 'use scheduler')
    parser.add_argument('--num-layers', default=3, type=int,
            help = 'number of b|t|m layers (overriden by layers if specified explicitly)')
    parser.add_argument('--layers', default='', type=str,
            help = 'specify hypergraph convolutional layers (b,t,m)')
    parser.add_argument('--bonds', default=True,
            help = 'whether to include bond hyperconv layers (only if layers are not explicitly specified)', action = 'store_true')
    parser.add_argument('--triplets', default=False,            help = 'whether to include triplet hyperconv layers (only if layers are not explicitly specified)', action = 'store_true')
    parser.add_argument('--motifs', default=False,
            help = 'whether to include motif hyperconv layers (only if layers are not explicitly specified)', action = 'store_true')
    parser.add_argument('--motif-feats', default=['csm','lsop'],
            help = 'type of motif feature included (csm or lsop)')

    args = parser.parse_args()

    layers = []
    if args.layers != '':
        a = 'model_'+args.layers
        for lay in args.layers:
            layers.append(str(lay))
        best_model_filename = f'best_{a}_{args.target_name}_{datetime.now()}.pth.tar'
        checkpoint_filename = f'checkpoint_{a}_{args.target_name}_{datetime.now()}.pth.tar'
    else:
        a = 'model_'
        if args.bonds:
            a += 'b'
        if args.triplets:
            a += 't'
        if args.motifs:
            a += 'm'
        a+=args.num_layers
        best_model_filename = f'best_{a}_{args.target_name}_{datetime.now()}.pth.tar'
        checkpoint_filename = f'checkpoint_{a}_{args.target_name}_{datetime.now()}.pth.tar'


    best_accu = 1e6
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)


    torch.backends.cudnn.benchmark = True
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="chgcnn",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "architecture": "Agg" ,
        "dataset": args.target_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size
        }
    )

    #### Create dataset
    print(f'Finding data for {matbench_task}...')
    processed_data_dir = f'data/dataset_{str(matbench_task.dataset_name)}'
    dataset = InMemoryCrystalHypergraphDataset(processed_data_dir)
    motif_feat_dim = 59
    train_index, targets = matbench_task.get_train_and_val_data(fold)
    train_indexes = list(train_index.index)
    test_indexes  = list(matbench_task.get_test_data(fold, include_target=False).index)
        
    #### Initiliaze model 
    print('Initializing model...') 
    if args.task == 'classification':
        class_bool = True
    else:
        class_bool = False

    model = CrystalHypergraphConv(classification = class_bool, bonds = args.bonds, motif_feat_dim = motif_feat_dim, triplets = args.triplets, motifs = args.motifs, layers = layers).to(device)

    
    #### Divide data into train and test sets
    dataset_train, dataset_val, dataset_test = train_val_test_data_from_indexes(dataset, train_indexes, test_indexes, args.train_ratio)

    #### Load data into DataLoaders/Batches
    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        pin_memory=False, #args.pin_memory,
        generator=torch.Generator().manual_seed(args.seed)
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        pin_memory=args.pin_memory
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        pin_memory=args.pin_memory
    )

    #### Set normalizer (for targets)
    if args.normalize == True:
        if len(dataset) < 1000:
            sample_targets = [torch.tensor(float(dataset[i].y)) for i in range(len(dataset))]
        else:
            sample_targets = [torch.tensor(float(dataset[i].y)) for i in sample(range(len(dataset)), 1000)]
        normalizer = Normalizer(sample_targets)
        print('Normalizer initialized!')
    else:
        normalizer = None
        print('No normalizer utilized in main file...')


    #### Set optimizer
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        print(f'SGD chosen as optimizer with lr {args.lr}, mom {args.momentum}, wd {args.weight_decay}')
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f'Adam chosen as optimizer with lr {args.lr}, wd {args.weight_decay}')

    else:
        raise ValueError('Optimizer must be SGD or Adam.')

    #### Set up scheduler
    if args.scheduler == True:
        #scheduler = CosineAnnealingLR(optimizer, T_max = 300)
        scheduler = MultiStepLR(optimizer, milestones = args.milestones, gamma=0.1)

    #### Set cost and loss functions 
    if args.task == 'regression':
        loss_criterion = torch.nn.MSELoss()
        accuracy_criterion = torch.nn.L1Loss()
        print('Using MSE accuracy and L1 for training loss')
    elif args.num_class == 2 and args.task != 'regression':
        loss_criterion = torch.nn.CrossEntropyLoss()
        accuracy_criterion = ClassificationAccuracy
        print('Using cross entropy for training loss and basic accuracy')
    else:
        loss_criterion = torch.nn.CrossEntropyLoss()
        accuracy_criterion = torch.nn.CrossEntropyLoss()
        print('Using cross entropy for training loss and accuracy')

    #### Resume mechanism
    if args.resume:
        print("=> loading checkpoint")
        checkpoint = torch.load(checkpoint_name)
        args.start_epoch = checkpoint['epoch'] + 1
        best_accu = checkpoint['best_accu']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    #### Loop through train and test for set number of epochs
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss, train_accu = train(model, device, train_loader, loss_criterion, accuracy_criterion, optimizer,
                                       epoch, args.task, 'target', normalizer, scheduler)
        val_accu = validate(model, device, val_loader, loss_criterion, accuracy_criterion, epoch, args.task, 'target', normalizer)
        test_accu = validate(model, device, test_loader, loss_criterion, accuracy_criterion, epoch, args.task, 'target', normalizer, test = True)

        is_best = val_accu < best_accu
        best_accu = min(val_accu, best_accu)

        scheduler.step()
        print(f'STEP: lr ={scheduler.get_last_lr()}')
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_accu': best_accu,
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
        }, is_best, checkpoint_filename, best_model_filename)



    ckpt = torch.load(best_model_filename)
    print(ckpt['best_accu'])

    test_outputs = validate(model, device, test_loader, loss_criterion, accuracy_criterion, epoch, args.task, 'target', normalizer, test = True, return_outs = True)

    wandb.finish()


    return test_outputs

#### Define normalizer for target data (from CGCNN source code)
class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor_list):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(torch.stack(tensor_list,dim=0))
        self.std = torch.std(torch.stack(tensor_list,dim=0))

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


def save_checkpoint(state, is_best, filename, best_model_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_model_filename)


if __name__ == '__main__':
    mb = MatbenchBenchmark(autoload=False, subset= [#'matbench_dielectric',
                                                    'matbench_log_gvrh',
                                                    'matbench_log_kvrh',
                                                    'matbench_perovskites',
                                                    'matbench_phonons', 
                                                    'matbench_mp_e_form',
                                                    'matbench_mp_gap',
                                                    'matbench_mp_is_metal'])


    for task in mb.tasks:
        task.load()
        for fold in task.folds:
            output = main(task, fold)
            task.record(fold, output)

    mb.to_file("my_models_benchmark_{datetime.now()}.json.gz")
