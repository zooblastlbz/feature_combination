import os
import sys
import json
import argparse
import torch

# ================= 路径设置 =================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, CLIPTokenizer, GemmaTokenizer, CLIPTextModelWithProjection, AutoModelForCausalLM

# 导入你的模型定义
from diffusion.configs import DiTConfig, FuseDiTConfig
from diffusion.models import DiT, FuseDiT, AdaFuseDiT
from diffusion.pipelines import DiTPipeline, FuseDiTPipeline, FuseDiTPipelineWithCLIP, AdaFuseDiTPipeline


def get_torch_dtype(dtype_str):
    dtype_map = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def clean_state_dict(state_dict):
    """
    清洗权重键名：
    1. 移除 'module.' (DDP/DeepSpeed 产生的)
    2. 移除 '_orig_mod.' (torch.compile 产生的)
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        # 循环去除前缀，防止出现 module.module. 的情况
        while new_k.startswith("module."):
            new_k = new_k[7:]
        while new_k.startswith("_orig_mod."):
            new_k = new_k[10:]
        new_state_dict[new_k] = v
    return new_state_dict


def load_deepspeed_file(file_path):
    """
    专门读取 mp_rank_00_model_states.pt 这种文件
    """
    print(f"Loading DeepSpeed checkpoint directly from file: {file_path}")
    
    # map_location="cpu" 至关重要，防止直接加载到显存爆显存
    checkpoint = torch.load(file_path, map_location="cpu")
    
    state_dict = None

    # DeepSpeed 保存的 .pt 通常是一个字典，结构如下:
    # {
    #    'module': OrderedDict(...),  <-- 我们要这个
    #    'optimizer': ...,
    #    'lr_scheduler': ...,
    #    'csr_tensor_module_names': ...
    # }
    if isinstance(checkpoint, dict):
        if "module" in checkpoint:
            print("Found 'module' key in checkpoint, extracting weights...")
            state_dict = checkpoint["module"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # 也许它就是纯权重的 state_dict
            state_dict = checkpoint
            
    return state_dict


def load_checkpoint_weights(path):
    """
    通用加载函数，支持 目录 或 直接指向文件
    """
    # 1. 如果用户直接指定了文件 (如 mp_rank_00_model_states.pt)
    if os.path.isfile(path):
        return load_deepspeed_file(path)
    
    # 2. 如果是目录，尝试查找
    print(f"Scanning checkpoint directory: {path}")
    candidates = [
        "mp_rank_00_model_states.pt", # DeepSpeed
        "pytorch_model/mp_rank_00_model_states.pt", # DeepSpeed with subdir
        "model.safetensors", 
        "pytorch_model.bin",
    ]
    
    target_file = None
    for fname in candidates:
        full_path = os.path.join(path, fname)
        if os.path.exists(full_path):
            target_file = full_path
            break
            
    if target_file:
        # 递归调用自己处理文件
        return load_checkpoint_weights(target_file)
    
    raise FileNotFoundError(f"Could not find valid model file in {path}")


def main(args):
    weight_dtype = get_torch_dtype(args.dtype)
    print(f"Target Dtype: {weight_dtype}")

    # ================= 1. 确定 Config 路径 =================
    # 如果 args.checkpoint 是文件，则 config 通常在它的父目录或者父目录的父目录
    config_path = args.checkpoint
    if os.path.isfile(config_path):
        # 回退一层: checkpoint-50000/pytorch_model/mp_rank.pt -> checkpoint-50000/pytorch_model
        config_path = os.path.dirname(config_path) 
        
    # 尝试寻找 config.json
    # 常见 DeepSpeed 结构: checkpoint-50000/config.json 
    # 而 pt 文件在 checkpoint-50000/pytorch_model/mp_rank.pt
    # 所以可能需要向上回退两层
    if not os.path.exists(os.path.join(config_path, "config.json")):
        parent_dir = os.path.dirname(config_path)
        if os.path.exists(os.path.join(parent_dir, "config.json")):
            config_path = parent_dir
            
    print(f"Loading config from: {config_path}")
    
    if args.type == "baseline-dit":
        config = DiTConfig.from_pretrained(config_path)
        model_cls = DiT
    elif "fuse-dit" in args.type:
        config = FuseDiTConfig.from_pretrained(config_path)
        model_cls = FuseDiT
    elif args.type == "adafusedit":
        config = DiTConfig.from_pretrained(config_path)
        model_cls = AdaFuseDiT
    else:
        raise ValueError(f"Unknown type: {args.type}")

    # ================= 2. 初始化模型 =================
    print("Initializing model architecture...")
    transformer = model_cls(config)

    # ================= 3. 加载权重 =================
    # 直接传入 args.checkpoint，现在支持文件路径了
    raw_state_dict = load_checkpoint_weights(args.checkpoint)
    
    # 清洗 Key (去除 module. 等前缀)
    print("Cleaning state dict keys...")
    state_dict = clean_state_dict(raw_state_dict)
    
    print("Applying state dict to model...")
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    
    if len(missing) > 0:
        print(f"[Warn] Missing keys ({len(missing)}): {missing[:3]} ...")
    if len(unexpected) > 0:
        print(f"[Warn] Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
        
    transformer.to(dtype=weight_dtype)

    # ================= 4. 处理 LLM/Tokenizer =================
    llm_path = args.llm_path
    if llm_path is None:
        if hasattr(config, "base_config") and hasattr(config.base_config, "_name_or_path"):
            llm_path = config.base_config._name_or_path
        elif hasattr(config, "_name_or_path"):
            llm_path = config._name_or_path
            
    print(f"Loading auxiliary models from: {llm_path}")
    
    tokenizer = None
    lm = None

    if llm_path:
        try:
            tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        except:
            tokenizer = GemmaTokenizer.from_pretrained(llm_path)
        try:
            from transformers import AutoModelForCausalLM
            lm = AutoModelForCausalLM.from_pretrained(model)
            return lm.model if hasattr(lm, "model") else lm
        except Exception:
            pass
        # 2) Try vision-language models and extract the language sub-module
        try:
            from transformers import AutoModelForImageTextToText
            vl = AutoModelForImageTextToText.from_pretrained(llm_path)
            for attr in ["language_model", "text_model", "model"]:
                if hasattr(vl, attr):
                    lm = getattr(vl, attr)
                    return lm.model if hasattr(lm, "model") else lm
            return vl
        except Exception as e:
            raise ValueError(f"Unknown model: {model}") from e
       

    if lm is not None:
        lm = lm.to(dtype=weight_dtype)

    # ================= 5. 构建 Pipeline =================
    print("Building Pipeline...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae")

    pipeline = None
    if args.type == "baseline-dit":
        pipeline = DiTPipeline(transformer=transformer, scheduler=scheduler, vae=vae, tokenizer=tokenizer, llm=lm)
    elif args.type == "adafusedit":
        pipeline = AdaFuseDiTPipeline(transformer=transformer, scheduler=scheduler, vae=vae, tokenizer=tokenizer, llm=lm)
    elif args.type == "fuse-dit":
        pipeline = FuseDiTPipeline(transformer=transformer, scheduler=scheduler, vae=vae, tokenizer=tokenizer)
    elif args.type == "fuse-dit-clip":
        clip = CLIPTextModelWithProjection.from_pretrained(args.clip_l, subfolder="text_encoder")
        clip_tok = CLIPTokenizer.from_pretrained(args.clip_l, subfolder="tokenizer")
        pipeline = FuseDiTPipelineWithCLIP(transformer=transformer, scheduler=scheduler, vae=vae, tokenizer=tokenizer, clip=clip, clip_tokenizer=clip_tok)

    # ================= 6. 保存 =================
    # 确定保存路径，如果是文件路径，则保存在文件所在的目录下
    save_base = args.checkpoint if os.path.isdir(args.checkpoint) else os.path.dirname(args.checkpoint)
    output_dir = os.path.join(save_base, "pipeline")
    
    print(f"Saving pipeline to: {output_dir}")
    pipeline.save_pretrained(output_dir)
    print("Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 这里直接传入你具体的 .pt 文件路径
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to mp_rank_00_model_states.pt")
    parser.add_argument("--type", type=str, default="fuse-dit", choices=["baseline-dit", "fuse-dit", "fuse-dit-clip", "adafusedit"])
    parser.add_argument("--llm_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    
    parser.add_argument("--scheduler", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--clip_l", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    
    args = parser.parse_args()
    main(args)