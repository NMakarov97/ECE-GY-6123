import argparse

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
import torchvision.utils as vision_utils
from tensorboardX import SummaryWriter

from data import getTrainingTestingData
from model import DenseDepth
from utils import AverageMeter, DepthNorm, colorize, load_from_checkpoint, init_or_load_model

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument(
        '--epochs',
        '-e',
        default=20,
        type=int,
        help='total number of epochs to run for training'
        )
    parser.add_argument(
        '--learning-rate',
        '--lr',
        default=0.0001,
        type=float,
        help='initial learning rate'
        )
    parser.add_argument(
        '--batch',
        '-b',
        default=8,
        type=int,
        help='batch size'
        )
    parser.add_argument(
        '--checkpoint',
        '-c',
        default='',
        type=str,
        help='path to last saved checkpoint to resume training'
        )
    parser.add_argument(
        '--resume_epoch',
        '-r',
        default=-1,
        type=int,
        help='epoch to resume training'
    )
    parser.add_argument(
        '--device',
        '-d',
        default='cuda',
        type=str,
        help='device to run training on'
    )
    parser.add_argument(
        '--enc_pretrain',
        '-p',
        default=True,
        type=bool,
        help='Use pretrained encoder'
    )
    parser.add_argument(
        '--data',
        default='nyu_depth_v2.zip',
        type=str,
        help='path to image dataset'
    )
    parser.add_argument(
        '--theta',
        '-t',
        default=0.1,
        type=float,
        help='coefficient for L1 (depth) loss'
    )
    parser.add_argument(
        '--save',
        '-s',
        default='',
        type=str,
        help='location to save checkpoints in'
    )
    args = parser.parse_args()

    # Set up various constants
    batch_size = args.batch
    model_prefix = 'DenseDepth_'
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    theta = args.theta
    save_count = 0
    epoch_loss = []
    batch_loss = []
    sum_loss = 0

    # Load data
    print('Loading data...')
    trainloader, testloader = getTrainingTestingData(args.data, batch_size=batch_size)
    num_trainloader = len(trainloader)
    num_testloader = len(testloader)
    print('Data loaders ready!')

    # Load from checkpoint if given
    if len(args.checkpoint) > 0:
        print('Loading from checkpoint...')
        model, optimizer, start_epoch = init_or_load_model(
            depthmodel=DenseDepth,
            enc_pretrain=args.enc_pretrain,
            epochs=args.epochs,
            lr=args.lr,
            ckpt=args.checkpoint,
            device=device
        )
        print(f'Resuming from epoch #{start_epoch}')
    # Initialize new model if no checkpoint present
    else:
        print('Initializing new model...')
        model, optimizer, start_epoch = init_or_load_model(
            depthmodel=DenseDepth,
            enc_pretrain=args.enc_pretrain,
            epochs=args.epochs,
            lr=args.lr,
            ckpt=None,
            device=device
        )

    # Set up logging
    writer = SummaryWriter(
        comment=f'{model_prefix}-learning_rate:{args.lr}-epoch:{args.epochs}-batch_size:{args.batch}'
    )

    # Loss functions
    l1_criterion = nn.L1Loss()
