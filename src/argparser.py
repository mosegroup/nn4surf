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



class MinimizerParser:
    # This class implements a simple minimization parser

    def __init__(self):

        self.parser = ArgumentParser() # composition with a regular parser

        self.parser.add_argument(
            '--output_folder',
            type        = str,
            default     = 'minimizations',
            help        = 'Folder in which minimization outputs will be stored'
            )

        self.parser.add_argument(
            '--name',
            type        = str,
            default     = 'min',
            help        = 'The specific minimization name'
            )

        self.parser.add_argument(
            '--shift',
            type        = float,
            default     = 0.0,
            help        = 'The heights of the profiles to be considered'
            )

        self.parser.add_argument(
            '--read_profile',
            action      = 'store_true',
            help        = 'Reload profile (specify path with "--path" argument)'
            )

        self.parser.add_argument(
            '--profile_path',
            type        = str,
            help        = 'Path of the profile to be reloaded (either .npy or .dat file)'
            )

        self.parser.add_argument(
            '--gaussian_area',
            type        = float,
            default     = 0.05,
            help        = 'The area of the Gaussian dimple on top of shift'
            )

        self.parser.add_argument(
            '--model_path',
            type        = str,
            default     = 'models/model.pt',
            help        = 'Trained model path'
            )

        self.parser.add_argument(
            '--device',
            type        = str,
            default     = 'cpu',
            help        = 'Device to be used ("cpu", "cuda", "cuda:#" on multi-GPU systems)'
            )

        self.parser.add_argument(
            '--alpha',
            default     = 5e-2,
            type        = float,
            help        = 'Alpha parameter in steepest descent minimization'
            )

        self.parser.add_argument(
            '--max_tol',
            type        = float,
            default     = 1e-4,
            help        = 'Tolerance on maximum derivative value in minimization'
            )

        self.parser.add_argument(
            '--eigen',
            type        = float,
            default     = 0.0399,
            help        = 'Eigenstrain to be used in "rescaling trick"'
            )

        self.parser.add_argument(
            '--steps_max',
            type        = int,
            default     = 300_000,
            help        = 'Maximum number of iterations in minimization'
            )



    def parse_args(self):
        # Parse argument and create folders

        args = self.parser.parse_args()

        if args.read_profile:
            if not args.profile_path.endswith('.npy') or not args.profile_path.endswith('.dat'):
                raise ValueError(f'The provided file for profile reloading "{args.profile_path}" does not have a .dat ot .npy extension.')

        if not torch.cuda.is_available() and args.device.startswith('cuda'):
            print('Cuda is not available on the current machine... falling back on cpu')
            args.device = 'cpu'

        if not os.path.isdir(args.output_folder):
            os.mkdir(args.output_folder)

        if os.path.isdir(f'{args.output_folder}/{args.name}'):
            raise NameError(f'Path "{args.output_folder}/{args.name}" is not emtpy. Please, clean-up position or change minimization name.')
        else:
            os.mkdir(f'{args.output_folder}/{args.name}')


        return args
        
