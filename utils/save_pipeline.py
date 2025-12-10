import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from io import BytesIO
import zstandard as zstd

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
import torch
from transformers import AutoTokenizer, CLIPTokenizer, GemmaTokenizer, AutoModel, CLIPTextModelWithProjection

from diffusion.configs import DiTConfig, FuseDiTConfig
from diffusion.models import DiT, FuseDiT, AdaFuseDiT
from diffusion.pipelines import DiTPipeline, FuseDiTPipeline, FuseDiTPipelineWithCLIP, AdaFuseDiTPipeline


def get_torch_dtype(dtype_str):
    """将字符串转换为 torch dtype"""
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def main(args):
    # 确定权重精度
    weight_dtype = get_torch_dtype(args.dtype)
    print(f"Using weight dtype: {weight_dtype}")
    
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
    elif args.trainer == "accelerate":
        checkpoint_dir = args.checkpoint
        safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
        bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"Could not find model.safetensors or pytorch_model.bin in {checkpoint_dir}")
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

    # 将 transformer 转换为目标精度，与训练代码保持一致
    transformer = transformer.to(dtype=weight_dtype)
    print(f"Transformer converted to {weight_dtype}")

    # Determine tokenizer/LLM path
    model_path = args.llm_path
    if model_path is None:
        model_path = config.base_config._name_or_path
    
    # Handle case where path is empty or invalid
    if not model_path:
        print("Warning: `_name_or_path` in config is empty and `--llm_path` was not provided.")
        print("Please provide the path to the LLM/Tokenizer using `--llm_path`.")
    
    print(f"Loading tokenizer and LLM from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load AutoTokenizer from {model_path}: {e}")
        print("Falling back to GemmaTokenizer...")
        tokenizer = GemmaTokenizer.from_pretrained(model_path)
        

    try:
        from transformers import AutoModelForCausalLM
        lm = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=weight_dtype)
        lm = lm.model if hasattr(lm, "model") else lm
    except Exception:
        lm = None

    if lm is None:
        # 2) Try vision-language models and extract the language sub-module
        try:
            from transformers import AutoModelForImageTextToText
            vl = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=weight_dtype)
            for attr in ["language_model", "text_model", "model"]:
                if hasattr(vl, attr):
                    lm = getattr(vl, attr)
                    # 对于 language_model，可能还需要取 .model
                    if hasattr(lm, "model"):
                        lm = lm.model
                    break
        except Exception as e:
            raise ValueError(f"Unknown model: {model_path}") from e

    # 确保 LLM 使用正确的精度
    if lm is not None:
        lm = lm.to(dtype=weight_dtype)
        print(f"LLM converted to {weight_dtype}")

    if args.type == "baseline-dit":
        pipeline = DiTPipeline(
            transformer=transformer,
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler, subfolder="scheduler"),
            vae=AutoencoderKL.from_pretrained(args.vae, subfolder="vae"),
            tokenizer=tokenizer,
            llm=lm,
        )
    elif args.type == "adafusedit":
        pipeline = AdaFuseDiTPipeline(
            transformer=transformer,
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler, subfolder="scheduler"),
            vae=AutoencoderKL.from_pretrained(args.vae, subfolder="vae"),
            tokenizer=tokenizer,
            llm=lm,
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
    print(f"Pipeline saved to {os.path.join(args.checkpoint, 'pipeline')}")


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
    parser.add_argument("--dtype", type=str, default="bfloat16", 
                        choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
                        help="Weight dtype for transformer and LLM (default: bfloat16, matching training)")
    args = parser.parse_args()
    main(args)