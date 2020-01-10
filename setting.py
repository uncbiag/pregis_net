import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_list',
        default='./data/ct_cbct/train_1.txt',
        type=str,
        help='Path for training image list file'
    )
    parser.add_argument(
        '--test_list',
        default='./data/ct_cbct/test_1.txt',
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
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    args = parser.parse_args()

    return args