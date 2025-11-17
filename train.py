import argparse
from omegaconf import OmegaConf

from diffusion.trainers import get_trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1) # only used for DeepSpeed
    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to DeepSpeed config JSON file")
    args = parser.parse_args()

    hparams = OmegaConf.load(args.config)
    trainer = get_trainer(hparams, args.local_rank, args.deepspeed_config)
    trainer.train()