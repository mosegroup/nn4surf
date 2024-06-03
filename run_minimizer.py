# <<< import external modules <<<
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import random
import time
# ==== import external modules ===

# <<< import nn4surf modules <<<
from src.classes import Minimizer
from src.physics import kappagamma, mu_wet
from src.argparser import MinimizerParser
from src.utils import init_profile_gaussian_dimple, reload_dat_profile, reload_npy_profile, save_args
# === import nn4surf modules ===

def main() -> None:
    # The main function!
    arg_parser = MinimizerParser()
    args = arg_parser.parse_args()

    save_args(f'{args.output_folder}/{args.name}/args.txt', args)

    x = np.arange(100) # define domain size (100 only for the moment)

    if args.read_profile:
        if args.profile_path.endswith('.npy'):
            prof = reload_npy_profile( args.path )
        elif args.profile_path.endswith('.dat'):
            prof = reload_dat_profile( args.path )
        #else cases should be already taken care by the parser
    else:
        print('Generating a new initial profile with Gaussian dimple')
        prof = init_profile_gaussian_dimple(x, args.gaussian_area, args.shift)
    
    prof_0 = prof.copy()
    
    # instantiate minimizer
    minimizer = Minimizer(
            args.model_path,
            analytic_terms = lambda h,dx : kappagamma(h,dx) + mu_wet(h,dx),
            device      = args.device
            )
    
    # perform minimization
    prof, mu_end = minimizer.minimize(
            prof,
            alpha       = args.alpha,
            max_tol     = args.max_tol,
            eigen       = args.eigen,
            steps_max   = args.steps_max
            )
    
    # save output
    np.savetxt(
        f'{args.output_folder}/{args.name}.dat',
        np.column_stack( (x, prof_0.squeeze(), prof.squeeze().cpu(), mu_end.squeeze().cpu()) ),
        header = 'x \t h_0 \t h'
        )


if __name__ == '__main__':
    main()
