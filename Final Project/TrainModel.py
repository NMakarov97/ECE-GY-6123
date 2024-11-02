import argparse
import time
import datetime
import os.path

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
import torchvision.utils as vision_utils
from tensorboardX import SummaryWriter

from data import getTrainingTestingData
from model import DenseDepth
from loss import ssim
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

    # Start training
    print(f'Device: {device}')
    print('Starting training...')

    for epoch in range(start_epoch, args.epochs):
        # Set up averaging
        batch_time = AverageMeter()
        losses = AverageMeter()

        # Switch to train mode
        model.train()
        model = model.to(device)
        epoch_start = time.time()
        end = time.time()

        for idx, batch in enumerate(trainloader):
            optimizer.zero_grad()

            # Prepare sample and target
            image = torch.Tensor(batch['image']).to(device)
            depth = torch.Tensor(batch['depth']).to(device)

            # Normalize depth
            normalized_depth = DepthNorm(depth)

            # Predict
            output = model(image)

            # Compute loss
            l1_loss = l1_criterion(output, normalized_depth)
            ssim_loss = torch.clamp(
                (1 - ssim(output, normalized_depth, 1000.0 / 10.0)) * 0.5,
                min=0,
                max=1
            )
            loss = (1.0 * ssim_loss) + (0.1 * l1_loss)

            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(num_trainloader-idx))))

            # Log progress
            num_iters = epoch * num_trainloader + idx
            if idx % 5 == 0:
                # Print status to console
                print(
                    f'Epoch: #{epoch} Batch: {idx}/{num_trainloader}\t'
                    f'Time (current/total) {batch_time.val:.3f}/{batch_time.sum:.3f}\t'
                    f'eta {eta}\t'
                    f'Loss (current/average) {losses.val:.4f}/{losses.avg:.4f}\t'
                )
                writer.add_scalar('Train/Loss', losses.val, num_iters)

            # Delete resources
            del image
            del depth
            del output

        # Save checkpoints
        if epoch % 1 == 0:
            print(
                '----------------------------------\n'
                f'Epoch: #{epoch}, Avg. Net Loss: {losses.avg:.4f}\n'
                '----------------------------------'
            )
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'loss': losses.avg
                },
                os.path.join(args.save, f'ckpt_{epoch}_{int(losses.avg * 100)}.pth')
            )
            LogProgress(model, writer, testloader, num_iters, device)
            writer.add_scalar('Train/Loss.avg', losses.avg, epoch)
