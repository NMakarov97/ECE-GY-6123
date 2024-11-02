import argparse

from data import getTrainingTestingData

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

    # Load data
    print('Loading data...')
    trainloader, testloader = getTrainingTestingData(args.data, batch_size=args.batch)
    num_trainloader = len(trainloader)
    num_testloader = len(testloader)
    print('Data loaders ready!')
