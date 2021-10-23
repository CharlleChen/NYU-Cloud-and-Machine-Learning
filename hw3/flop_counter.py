from ptflops import get_model_complexity_info
import argparse
from main import Net
import torch

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--inc-conv', action='store_true', default=False,
                    help='change the kernel size of the convolutional layer')
parser.add_argument('--inc-linear', action='store_true', default=False,
                    help='change the kernel size of the linear layer')
parser.add_argument('--mark-layer', action='store_true', default=False,
                    help='whether profile the whole process')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")
# .to(device)

with torch.cuda.device(0):
  model = Net(args)
  macs, params = get_model_complexity_info(model, (1, 28, 28), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))
