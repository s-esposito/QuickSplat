import argparse
import random
from datetime import timedelta
from pathlib import Path
import shutil

import numpy as np
import torch

from configs.config import get_cfg_defaults
from trainers.inference_trainer import InferenceTrainer
from utils.rich_utils import CONSOLE


DEFAULT_TIMEOUT = timedelta(minutes=30)


def parse_args():
    # Get config file path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--out", type=str, required=True, help="Path to output folder")
    # Other arguments (any input) that will be later merged into yacs config
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Modify config options using the command-line")
    return parser.parse_args()


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True  # type: ignore


def main():
    args = parse_args()
    config = get_cfg_defaults()
    config.merge_from_file(args.config)
    config.merge_from_list(args.opts)

    # Extract config filename without extension to use as directory name
    config_path = Path(args.config)
    config_name = config_path.stem  # Gets filename without extension
    
    # Use config name as the subname for output directory
    config.subname = config_name
    
    config.log_dir = args.out
    config.save_dir = args.out
    config.do_logging = False

    config.freeze()
    _set_random_seed(config.MACHINE.seed)
    
    trainer = InferenceTrainer(
        config,
        ckpt_path=args.ckpt,
        local_rank=0,
        world_size=1,
    )
    
    # Copy config file to output directory
    output_dir = Path(config.save_dir) / config.name / config.subname
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dest = output_dir / config_path.name
    shutil.copy2(args.config, config_dest)
    print(f"Copied config file to: {config_dest}")
    
    trainer.train()


if __name__ == "__main__":
    main()
