import os
import subprocess
import torch
import shutil
import yaml
osp = os.path

from sinf.inrs import fit_3d, mlp
from sinf.utils import util

from sinf import RESULTS_DIR, TMP_DIR

def launch_job(config_name, subject='', prefix='man'):
    if config_name.startswith('fet'):
        job_name = prefix+subject[3:]
    elif config_name.startswith('echo'):
        job_name = prefix+subject[-3:]
    elif config_name.startswith('nsd'):
        job_name = prefix+subject[-3:]
    subprocess.run(osp.join(osp.expandvars('$NFS'), f'/code/sinf/scripts/fit_inr.sh {job_name} {config_name} {subject}'), shell=True)

def rename_job(job: str, new_name: str):
    os.rename(osp.join(RESULTS_DIR, job), osp.join(RESULTS_DIR, new_name))
    for folder in util.glob2(TMP_DIR, "*", job):
        os.rename(folder, folder.replace(job, new_name))

def delete_job(job: str):
    shutil.rmtree(osp.join(RESULTS_DIR, job))
    for folder in util.glob2(TMP_DIR, "*", job):
        shutil.rmtree(folder)

def get_job_args(job: str) -> dict:
    config_path = osp.join(RESULTS_DIR, job, f"config.yaml")
    args = yaml.safe_load(open(config_path, "r"))
    return args

def get_dataset_for_job(job: str):
    return get_job_args(job)["data loading"]["dataset"]

def load_model_for_job(job_id: str):
    kwargs = get_job_args(job_id)
    nf = fit_3d.construct_model(kwargs, (112, 112, 80))
    nf.load_state_dict(torch.load(kwargs['paths']["job output dir"]+f'/weights_{job_id}.pt'))
    return nf.eval()