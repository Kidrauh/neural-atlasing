from typing import Optional, List
import yaml, wandb

from sinf import CONFIG_DIR

class Sweep:
    def __init__(self, name:str, sweeps:Optional[List]=None):
        self.name = name
        self.sweeps = sweeps

def get_param_dict(vmin, vmax, distribution='log_uniform_values'):
    return {
        'distribution': distribution,
        'min': vmin,
        'max': vmax,
    }

def create_sweep(name, method="random", target='val_loss', goal='minimize', parameter_dict=None):
    if parameter_dict is None:
        parameter_dict = yaml.safe_load(open(f'{CONFIG_DIR}/sweeps/{name}.yaml', 'r'))
    for p in parameter_dict:
        if "distribution" not in parameter_dict[p]:
            parameter_dict[p] = get_param_dict(parameter_dict[p]['min'], parameter_dict[p]['max'])

    sweep_config = {
        "name": name,
        "method": method,
        'metric': {
            'name': target,
            'goal': goal,
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 100,
        },
        "parameters": parameter_dict,
    }
    sweep_id = wandb.sweep(sweep_config, project='sinf')
    # return sweep_id
    experiment = Sweep(name, sweeps=[sweep_id])
    return experiment

def sweep_discretization_type():
    sweep_config = {
        "name": "sweep discretization_type",
        "method": "grid",
        "parameters": {
            "discretization_type": {
                "values": ["rqmc", 'grid', "shrunk"],
            },
        }
    }
    return wandb.sweep(sweep_config, project='sinf')

def sweep_parameter(parameter_name, values):
    name = f'{parameter_name}_sweep'
    sweep_config = {
        "name": name,
        "method": "grid",
        "parameters": {
            parameter_name: {
                "values": values,
            },
        }
    }
    sweep_id = wandb.sweep(sweep_config, project='sinf')
    experiment = Sweep(name, sweeps=[sweep_id])
    return experiment

