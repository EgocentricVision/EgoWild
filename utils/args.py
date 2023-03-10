import os
import platform
from datetime import datetime

from omegaconf import OmegaConf

# Retrieve the configs path
conf_path = os.path.join(os.path.dirname(__file__), '../configs')

# Read the cli args
cli_args = OmegaConf.from_cli()

# read a specific config file
args = OmegaConf.create()
if 'config' in cli_args and cli_args.config:
    base_conf_path = cli_args.config
else:
    base_conf_path = "debug.yaml"

while base_conf_path is not None:
    base_conf = OmegaConf.load(os.path.join(conf_path, base_conf_path))
    base_conf_path = base_conf.extends
    args = OmegaConf.merge(base_conf, args)

# try loading environment-specific configuration
try:
    machine = platform.node()
    path_args = OmegaConf.load(f"env_configs/{machine}.yaml")
    args = OmegaConf.merge(args, path_args)
except:
    pass  # no environment-specific configuration provided

# Merge cli args into config ones
args = OmegaConf.merge(args, cli_args)

# add log directories
args.experiment_dir = os.path.join(args.name, datetime.now().strftime('%b%d_%H-%M-%S'))
if args.action != "train":
    args.log_dir = os.path.join('TEST_RESULTS', args.name)
    if args.logname is None:
        args.logname = args.action + "_" + args.dataset.shift + ".log"
    else:
        args.logname = args.logname + "_" + args.dataset.shift + ".log"
    args.logfile = os.path.join(args.log_dir, args.logname)
else:
    args.log_dir = os.path.join('Experiment_logs', args.experiment_dir)
    args.logfile = os.path.join(args.log_dir, args.action + ".log")
os.makedirs(args.log_dir, exist_ok=True)
if args.models_dir is None:
    args.models_dir = "saved_models"
if args.action != "train" and args.action != 'save' and args.resume_from is None:
    if args.resume_name is None:
        resume_name = args.name
    else:
        resume_name = args.resume_name
    args.resume_from = os.path.join(args.models_dir, resume_name)
