import argparse

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

if __name__ == '__main__':
    main()
