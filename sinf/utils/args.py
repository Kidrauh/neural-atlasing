"""
Argument parsing
"""
import argparse
import os
import shutil
import wandb
import yaml
from glob import glob
osp = os.path

from sinf import CONFIG_DIR

def parse_args(args):
    """Command-line args for fit_inr.sh"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_name')
    parser.add_argument('-s', '--subject_id', default=None)
    parser.add_argument('-j', '--job_id', default="manual")
    parser.add_argument('-w', '--sweep_id', default=None)
    parser.add_argument('-t', '--start_ix', default=0, type=int)
    cmd_args = parser.parse_args(args)

    config_name = cmd_args.job_id if cmd_args.config_name is None else cmd_args.config_name
    configs = glob(osp.join(CONFIG_DIR, "*", config_name+".yaml"))
    if len(configs) == 0:
        raise ValueError(f"Config {config_name} not found in {CONFIG_DIR}")
    args = args_from_file(configs[0], cmd_args)
    os.makedirs(args['paths']["job output dir"], exist_ok=True)

    return args

def get_wandb_train_config():
    wandb_sweep_dict = {
        'learning_rate': ['optimizer', 'learning rate'],
        'batch_size': ['data loading', 'batch size'],
        'sample_points': ['data loading', 'sample points'],
        'discretization_type': ['data loading', 'discretization type'],
        'augment': ['data loading', 'augment'],
        'weight_decay': ['optimizer', 'weight decay'],
        'optimizer_type': ['optimizer', 'type'],

        'num_layers': ['network', 'depth'],
        'num_channels': ['network', 'channels'],
        # 'inter_grid_size': ['network', 'intermediate grid size'],
        
        'kernel_size_0': ['network', 'conv', 'k0'],
        'kernel_size_1': ['network', 'conv', 'k1'],
        'kernel_size_2': ['network', 'conv', 'k2'],
        'kernel_size_3': ['network', 'conv', 'k3'],
        'posenc_order': ['network', 'conv', 'posenc_order'],
        'pe_scale': ['network', 'conv', 'pe_scale'],
        'conv_mlp_type': ['network', 'conv', 'mlp_type'],
        'conv_N_bins': ['network', 'conv', 'N_bins'],
        'conv_N_ch': ['network', 'conv', 'mid_ch'],
    }
    if wandb.config['sweep_id'] is not None:
        for k,subkeys in wandb_sweep_dict.items():
            if k in wandb.config:
                d = wandb.config
                for subk in subkeys[:-1]:
                    d = d[subk]
                d[subkeys[-1]] = wandb.config[k]
    wandb.config.persist()
    args = dict(wandb.config.items())
    yaml.safe_dump(args, open(osp.join(args['paths']["job output dir"], "config.yaml"), 'w'))
    return args

def get_wandb_fit_config():
    # sweeps can only specify top-level args
    wandb_sweep_dict = {
        'learning_rate': ['optimizer', 'learning rate'],
        'weight_decay': ['optimizer', 'weight decay'],
        'n_deform_layers': ['deformation', 'network_config', 'n_hidden_layers'],
        'n_static_features': ['static', 'encoding_config', 'nested', 0, 'n_features_per_level'],
    }

    if wandb.config['sweep_id'] is not None:
        for k,subkeys in wandb_sweep_dict.items():
            if k in wandb.config:
                d = wandb.config
                for subk in subkeys[:-1]:
                    d = d[subk]
                d[subkeys[-1]] = wandb.config[k]
    wandb.config.persist()
    args = dict(wandb.config.items())

    yaml.safe_dump(args, open(osp.join(args['paths']["job output dir"], "config.yaml"), 'w'))
    return args

def merge_args(parent_args, child_args):
    """Merge parent config args into child configs."""
    if "_overwrite_" in child_args.keys():
        return child_args
    for k,parent_v in parent_args.items():
        if k not in child_args.keys():
            child_args[k] = parent_v
        else:
            if isinstance(child_args[k], dict) and isinstance(parent_v, dict):
                child_args[k] = merge_args(parent_v, child_args[k])
    return child_args


def args_from_file(path, cmd_args=None):
    """Create args dict from yaml."""
    if osp.exists(path):
        args = yaml.safe_load(open(path, 'r'))
    else:
        raise ValueError(f"bad config_name {cmd_args.config_name}")

    if cmd_args is not None:
        for param in ["job_id", "config_name", 'sweep_id', 'no_wandb', 'start_ix', 'subject_id']:
            if hasattr(cmd_args, param):
                args[param] = getattr(cmd_args, param)

    while "parent" in args:
        if isinstance(args["parent"], str):
            config_path = glob(osp.join(CONFIG_DIR, "*", args.pop("parent")+".yaml"))[0]
            args = merge_args(yaml.safe_load(open(config_path, 'r')), args)
        else:
            parents = args.pop("parent")
            for p in parents:
                config_path = glob(osp.join(CONFIG_DIR, "*", p+".yaml"))[0]
                args = merge_args(yaml.safe_load(open(config_path, 'r')), args)
            if "parent" in args:
                raise NotImplementedError("need to handle case of multiple parents each with other parents")

    config_path = osp.join(CONFIG_DIR, "default.yaml")
    args = merge_args(yaml.safe_load(open(config_path, 'r')), args)
    infer_missing_args(args)
    return args

def infer_missing_args(args):
    """Convert str to float, etc."""
    if args['subject_id'] in (None, 'None'):
        raise ValueError("subject_id must be specified")
    paths = args["paths"]
    paths["job output dir"] = osp.join(osp.expandvars(paths["slurm output dir"]), args["job_id"])
    paths["weights dir"] = osp.join(paths["job output dir"], "weights")
    for k in args["optimizer"]:
        if isinstance(args["optimizer"][k], str) and args["optimizer"][k].startswith("1e"):
            args["optimizer"][k] = float(args["optimizer"][k])
    for k in args["scheduler"]:
        if isinstance(args["scheduler"][k], str) and args["scheduler"][k].startswith("1e"):
            args["scheduler"][k] = float(args["scheduler"][k])
