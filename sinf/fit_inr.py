import os
import sys
import numpy as np
import torch
import wandb

osp = os.path
from functools import partial

from sinf.utils import args as args_module
from sinf.inrs import fit_2d, fit_3d
from sinf.experiments import sweep

def main():
    os.environ['WANDB_API_KEY'] = open(os.path.expandvars('$NFS/.wandb'), 'r').read().strip()
    args = args_module.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    if args.get("random seed", -1) >= 0:
        np.random.seed(args["random seed"])
        torch.manual_seed(args["random seed"])
    method = fit_3d.main

    if args.get('sweep_id', True):
        wandb.agent(args['sweep_id'], function=partial(method, args=args), count=1, project='sinf')
    else:
        method(args=args)


if __name__ == "__main__":
    main()
