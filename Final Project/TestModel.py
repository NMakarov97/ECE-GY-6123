import argparse
from os import path, listdir

import torch

from model import DenseDepth
from utils import init_or_load_model

def main() -> None:
    # Command line arguments
    parser = argparse.ArgumentParser(description='Monocular Depth Estimation Testing')
    parser.add_argument(
        '--checkpoint',
        '-c',
        type=str,
        help='path to checkpoint'
    )
    parser.add_argument(
        '--device',
        '-d',
        default='cuda',
        type=str,
        help='device to run testing with'
    )
    parser.add_argument(
        '--data',
        default=R'Final Project\testing',
        type=str,
        help='path to image dataset'
    )
    parser.add_argument(
        '--colorbar',
        default='plasma',
        type=str,
        help='color scheme to use for the results'
    )
    args = parser.parse_args()

    # Set up various constants
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')

    # Check checkpoint file
    if not path.isfile(args.checkpoint):
        raise NotADirectoryError(f'{args.checkpoint} is not a valid checkpoint file')

    # Load data
    print('Loading data...')
    images = [path.join(args.data, f) for f in listdir(args.data) if path.isfile(path.join(args.data, f)) and '.png' in f]
    print(f'Loaded {len(images)} images')

    # Load model from checkpoint for testing
    print('Loading from checkpoint...')
    model, _, _ = init_or_load_model(
        depthmodel=DenseDepth,
        enc_pretrain=False,
        epochs=0,
        lr=0.001,
        ckpt=args.checkpoint,
        device=device
    )
    model.eval()
    print('Model loaded from checkpoint!')

if __name__ == '__main__':
    main()
