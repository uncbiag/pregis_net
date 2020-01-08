import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_list',
        default='./data/sample_train.txt',
        type=str,
        help='Path for training image list file'
    )
    parser.add_argument(
        '--test_list',
        default='./data/sample_test.txt',
        type=str,
        help='Path for testing image list file'
    )
    parser.add_argument(
        '--input_D',
        default=128,
        type=int,
        help='Input size of depth')
    parser.add_argument(
        '--input_H',
        default=128,
        type=int,
        help='Input size of height')
    parser.add_argument(
        '--input_W',
        default=128,
        type=int,
        help='Input size of width')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Path for resume model'
    )
    args = parser.parse_args()

    return args