import argparse
import os

import torch

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
    if not os.path.isfile(args.checkpoint):
        raise NotADirectoryError(f'{args.checkpoint} is not a valid checkpoint file')

if __name__ == '__main__':
    main()
