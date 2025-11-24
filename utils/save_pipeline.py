import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from io import BytesIO
import zstandard as zstd

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
import torch
from transformers import AutoTokenizer, CLIPTokenizer, GemmaTokenizer, AutoModel, CLIPTextModelWithProjection, AutoConfig

from diffusion.configs import DiTConfig, FuseDiTConfig
from diffusion.models import DiT, FuseDiT, AdaFuseDiT, get_llm
from diffusion.pipelines import DiTPipeline, FuseDiTPipeline, FuseDiTPipelineWithCLIP, AdaFuseDiTPipeline


def main(args):
    if args.trainer == "deepspeed":
        checkpoint_dir = args.checkpoint
        tag = args.tag
        
        if tag is None:
            latest_path = os.path.join(checkpoint_dir, 'latest')
            if not os.path.isfile(latest_path):
                # If 'latest' file is missing, assume checkpoint_dir is the full path to the specific checkpoint
                # We need to split it because get_fp32_state_dict_from_zero_checkpoint expects (base, tag)
                checkpoint_dir, tag = os.path.split(checkpoint_dir.rstrip(os.sep))

        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)
    elif args.trainer == "spmd":
        if args.compression:
            with open(f"{args.checkpoint}/model.pt.zst", "rb") as f:
                compressed = f.read()
            data = zstd.decompress(compressed)
            state_dict = torch.load(BytesIO(data))
        else:
            state_dict = torch.load(f"{args.checkpoint}/model.pt")

        for k in list(state_dict.keys()):
            if "_orig_module." in k:
                state_dict[k.replace("_orig_module.", "")] = state_dict[k]
                del state_dict[k]
    else:
        raise ValueError(f"Unknown trainer: {args.trainer}")

    if args.type == "baseline-dit":
        config = DiTConfig.from_pretrained(args.checkpoint)
        transformer = DiT(config)
        transformer.load_state_dict(state_dict)
    elif "fuse-dit" in args.type:
        config = FuseDiTConfig.from_pretrained(args.checkpoint)
        transformer = FuseDiT(config)
        transformer.load_state_dict(state_dict)
    elif args.type == "adafusedit":
        config = DiTConfig.from_pretrained(args.checkpoint)
        transformer = AdaFuseDiT(config)
        transformer.load_state_dict(state_dict)

    # Determine tokenizer/LLM path
    model_path = args.llm_path
    if model_path is None:
        model_path = config.base_config._name_or_path
    
    # Handle case where path is empty or invalid
    if not model_path:
        print("Warning: `_name_or_path` in config is empty and `--llm_path` was not provided.")
        print("Please provide the path to the LLM/Tokenizer using `--llm_path`.")
        # We can't proceed reliably without a path, but let's try to let the user know.
        # If we continue, from_pretrained('') will fail.
    
    print(f"Loading tokenizer and LLM from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load AutoTokenizer from {model_path}: {e}")
        print("Falling back to GemmaTokenizer...")
        tokenizer = GemmaTokenizer.from_pretrained(model_path)

    # Load LLM using the same logic as training
    if args.type in ["baseline-dit", "adafusedit"]:
        # Ensure base_config is a Config object for get_llm checks
        if isinstance(config.base_config, dict):
            base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        else:
            base_config = config.base_config
        
        llm = get_llm(model_path, base_config)
    else:
        llm = None

    if args.type == "baseline-dit":
        pipeline = DiTPipeline(
            transformer=transformer,
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler, subfolder="scheduler"),
            vae=AutoencoderKL.from_pretrained(args.vae, subfolder="vae"),
            tokenizer=tokenizer,
            llm=llm,
        )
    elif args.type == "adafusedit":
        pipeline = AdaFuseDiTPipeline(
            transformer=transformer,
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler, subfolder="scheduler"),
            vae=AutoencoderKL.from_pretrained(args.vae, subfolder="vae"),
            tokenizer=tokenizer,
            llm=llm,
        )
    elif args.type == "fuse-dit":
        pipeline = FuseDiTPipeline(
            transformer=transformer,
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler, subfolder="scheduler"),
            vae=AutoencoderKL.from_pretrained(args.vae, subfolder="vae"),
            tokenizer=tokenizer
        )
    elif args.type == "fuse-dit-clip":
        pipeline = FuseDiTPipelineWithCLIP(
            transformer=transformer,
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler, subfolder="scheduler"),
            vae=AutoencoderKL.from_pretrained(args.vae, subfolder="vae"),
            tokenizer=tokenizer,
            clip=CLIPTextModelWithProjection.from_pretrained(args.clip_l, subfolder="text_encoder"),
            clip_tokenizer=CLIPTokenizer.from_pretrained(args.clip_l, subfolder="tokenizer"),
        )
    else:
        raise ValueError(f"Unknown type: {args.type}")
    
    pipeline.save_pretrained(os.path.join(args.checkpoint, "pipeline"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./output")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--trainer", type=str, default="deepspeed")
    parser.add_argument("--type", type=str, default="fuse-dit")
    parser.add_argument("--scheduler", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--clip_l", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--clip_g", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--t5", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--llm_path", type=str, default=None, help="Path to the LLM/Tokenizer")
    parser.add_argument("--compression", action="store_true")
    args = parser.parse_args()
    main(args)