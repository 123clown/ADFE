import os
import torch
import pprint
import argparse
from train import ADFF_main



os.environ["CUDA_VISIBLE_DEVICES"] = '0'


data_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset/nutrition5k_dataset')
loss_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/loss_image')
test_results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/test')
net_dump_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/dump')
pre_dump_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'swin_base_patch4_window12_384.pth')

def parse_args():
    desc = 'rgbd-transformer-nutrition estimation on nutrition5k'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--b', type=int, default=8, help='batch_size')
    parser.add_argument('--data_root', type=str, default=data_root, help='dataset root')
    parser.add_argument('--seed', type=int, default=1, help='random seed for initialization')
    parser.add_argument('--beta1', type=float, default=0.9, help='adam optimizer parameter')
    parser.add_argument('--beta2', type=float, default=0.999, help='adam optimizer parameter')
    parser.add_argument('--test', action='store_true', help='use test.txt and test image')
    parser.add_argument('--epoch', type=int, default=150, help='the number of train epoch')
    parser.add_argument('--cpu', action='store_true', help='if set, use cpu only')
    parser.add_argument('--load', type=str, default='', help='path to load network weights(if non-empty)')
    parser.add_argument('--loss_path', type=str, default=loss_path, help='loss image save path')
    parser.add_argument('--num_workers', type=int, default=0, help='total thread of data loader')
    parser.add_argument('--dump', type=str, default=net_dump_path, help='every epoch net_dump file save path')
    parser.add_argument('--t_result', type=str, default=test_results_path, help='test result save path')
    parser.add_argument('--pre', type=str, default=pre_dump_path, help='load pre-trained network')
    parser.add_argument('--nopre', action='store_true', help='not use pretrained resnet')
    parser.add_argument('--size', type=int, default=384, help='resize image ')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight_decay')
    
    
    
    args = parser.parse_args()
    validate_args(args)

    return args


def validate_args(args):
    print('validating arguments...')
    pprint.pprint(args.__dict__)

    assert args.epoch >= 1, 'number epochs must be larger than or equal to one'
    assert args.b >= 1, 'batch_size number must be larger than or equal to one'

    assert os.path.exists(args.data_root), 'cannot find train/test root directory'

def main():
    args = parse_args()
    if not args.cpu and args.seed < 0:
        torch.backends.cudnn.benchmark = True

    Net = ADFF_main(args)

    if args.test :
        Net.test()
        print("Test Finished!")
    else:
        Net.train()
        print('Training finished!')

if __name__ == '__main__':
    main()
