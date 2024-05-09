# <<< importing stuff <<<
import os
import torch
from argparse import ArgumentParser
# --- importing stuff ---

class Parser():
    '''
    This is a class implementing a parser reading command line options and defaulting to values used in the paper.
    '''

    def __init__(self):

        self.parser = ArgumentParser()

        self.parser.add_argument(
            '--output_folder',
            type        = str,
            default     = 'training_logs',
            help        = 'Folder in which training outputs (models included) will be stored'
            )

        self.parser.add_argument(
            '--device',
            type        = str,
            default     = 'cuda',
            help        = 'Device to be used for training ("cpu", "cuda" or "cuda:N" for machines with multiple GPUs)'
            )

        self.parser.add_argument(
            '--kernel_size',
            type        = int,
            default     = 21,
            help        = 'Receptive field for every convolution block'
            )

        self.parser.add_argument(
            '--depth',
            type        = int,
            default     = 5,
            help        = 'Number of convolution/tanh blocks'
            )

        self.parser.add_argument(
            '--channels',
            type        = int,
            default     = 20,
            help        = 'Number of convolutions for every conv block'
            )

        self.parser.add_argument(
            '--lr',
            type        = float,
            default     = 1e-5,
            help        = 'Learning Rate'
            )

        self.parser.add_argument(
            '--model_path',
            type        = str,
            default     = 'None',
            help        = 'Model to be realoaded path. Set to "None" to start from a fresh one.'
            )

        self.parser.add_argument(
            '--train_set',
            type        = str,
            default     = 'train.txt',
            help        = 'Training set table path'
            )

        self.parser.add_argument(
            '--valid_set',
            type        = str,
            default     = 'valid.txt',
            help        = 'Validation set table path'
            )

        self.parser.add_argument(
            '--nproc',
            type        = int,
            default     = 0,
            help        = 'Number of processes for dataloading parallelization'
            )

        self.parser.add_argument(
            '--no_graphics',
            action      = 'store_true',
            help        = 'Suppress graphical output of the pedicted values of elastic energy density'
            )

        self.parser.add_argument(
            '--debug',
            action      = 'store_true',
            help        = 'Toggles debug mode (training and validation will break after few iterations)'
            )

        self.parser.add_argument(
            '--replicas',
            type        = int,
            default     = 0,
            help        = 'Number of periodic replicas to be considered'
            )

        self.parser.add_argument(
            '--interpolation_mode',
            type        = str,
            default     = 'adaptive',
            help        = 'Set interpolation mode for different scales. "adaptive" uses (transposed)convolutions while "simple" uses bilinear interpolation'
            )

        


    def parse_args(self):
        '''
        Parsing argument method
        '''
        args = self.parser.parse_args()
        if not torch.cuda.is_available() and args.device.startswith('cuda'):
            print('Cuda is not available on the current machine... falling back on cpu')
            args.device = 'cpu'
        if args.no_graphics:
            args.graphics = False
        else:
            args.graphics = True

        return args
