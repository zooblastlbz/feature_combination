import os
import sys
import json
import argparse
import torch

# ================= 路径设置 =================
# 将上级目录添加到 path 以导入项目模块
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
    3. 移除 'shadow_params.' (部分 EMA 实现产生的)
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        # 循环去除前缀
        while new_k.startswith("module."):
            new_k = new_k[7:]
        while new_k.startswith("_orig_mod."):
            new_k = new_k[10:]
        new_state_dict[new_k] = v
    return new_state_dict


def load_generic_pt_file(file_path):
    """
    通用读取 .pt 文件的函数
    支持 DeepSpeed checkpoint (mp_rank_00...) 和 EMA checkpoint (ema.pt)
    """
    print(f"Loading checkpoint directly from file: {file_path}")
    
    # map_location="cpu" 防止爆显存
    checkpoint = torch.load(file_path, map_location="cpu")
    
    state_dict = None

    # .pt 文件通常是一个字典，我们需要找到包含权重的那个 key
    if isinstance(checkpoint, dict):
        # 1. 检查常见的 EMA 键名
        if "shadow_params" in checkpoint:
            print("Found 'shadow_params' (EMA) key in checkpoint...")
            state_dict = checkpoint["shadow_params"]
        # 2. 检查常见的 DeepSpeed/DDP 键名
        elif "module" in checkpoint:
            print("Found 'module' key in checkpoint...")
            state_dict = checkpoint["module"]
        # 3. 检查标准的 state_dict
        elif "state_dict" in checkpoint:
            print("Found 'state_dict' key in checkpoint...")
            state_dict = checkpoint["state_dict"]
        # 4. 如果字典里全是 tensor，那它本身就是 state_dict
        elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
             print("Checkpoint is a raw state dict.")
             state_dict = checkpoint
        else:
            print("[Warning] Could not identify explicit key. Trying to use dict as is.")
            state_dict = checkpoint
            
    return state_dict


def load_checkpoint_weights(path, use_ema=False):
    """
    通用加载函数
    :param path: 文件路径或目录路径
    :param use_ema: 如果是目录，是否优先加载 ema.pt
    """
    # 1. 如果用户直接指定了文件 (例如 .../ema.pt)
    if os.path.isfile(path):
        return load_generic_pt_file(path)
    
    # 2. 如果是目录，根据优先级查找
    print(f"Scanning checkpoint directory: {path}")
    
    # 定义候选文件列表
    if use_ema:
        # 如果指定了用 EMA，把 ema.pt 放在最前面
        candidates = [
            "ema.pt",
            "ema_state_dict.pt",
            # 备选：如果没找到 ema，回退到普通权重
            "mp_rank_00_model_states.pt", 
            "model.safetensors", 
            "pytorch_model.bin",
        ]
        print("-> Mode: Prefer EMA weights")
    else:
        candidates = [
            "mp_rank_00_model_states.pt",
            "pytorch_model/mp_rank_00_model_states.pt",
            "model.safetensors", 
            "pytorch_model.bin",
            "ema.pt" # 放在最后作为备选
        ]
    
    target_file = None
    for fname in candidates:
        full_path = os.path.join(path, fname)
        if os.path.exists(full_path):
            target_file = full_path
            print(f"Found match: {fname}")
            break
            
    if target_file:
        return load_checkpoint_weights(target_file, use_ema)
    
    raise FileNotFoundError(f"Could not find valid model file in {path}. Candidates checked: {candidates}")


def main(args):
    weight_dtype = get_torch_dtype(args.dtype)
    print(f"Target Dtype: {weight_dtype}")

    # ================= 1. 确定 Config 路径 =================
    config_path = args.checkpoint
    if os.path.isfile(config_path):
        # 如果是指向文件 (如 ema.pt)，回退一层到 checkpoint 目录
        config_path = os.path.dirname(config_path) 
        
    # 如果 config 不在 checkpoint 根目录，尝试向上寻找
    if not os.path.exists(os.path.join(config_path, "config.json")):
        parent_dir = os.path.dirname(config_path)
        if os.path.exists(os.path.join(parent_dir, "config.json")):
            config_path = parent_dir
            
    print(f"Loading config from: {config_path}")
    
    try:
        if args.type == "dit":
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
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # ================= 2. 初始化模型 =================
    print(f"Initializing model architecture: {model_cls.__name__}...")
    transformer = model_cls(config)

    # ================= 3. 加载权重 =================
    # 传入 use_ema 参数
    raw_state_dict = load_checkpoint_weights(args.checkpoint, use_ema=args.use_ema)
    
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
            print("AutoTokenizer failed, trying GemmaTokenizer...")
            tokenizer = GemmaTokenizer.from_pretrained(llm_path)
        
        # 尝试加载 LLM
        try:
            from transformers import AutoModelForCausalLM
            temp_lm = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=weight_dtype)
            if hasattr(temp_lm, "model") and isinstance(temp_lm.model, torch.nn.Module):
                lm = temp_lm.model
            else:
                lm = temp_lm
        except Exception:
            lm = None

        # 如果失败，尝试作为 VLM 加载
        if lm is None:
            try:
                from transformers import AutoModelForImageTextToText
                vl = AutoModelForImageTextToText.from_pretrained(llm_path, torch_dtype=weight_dtype)
                for attr in ["language_model", "text_model", "model"]:
                    if hasattr(vl, attr):
                        lm_part = getattr(vl, attr)
                        if hasattr(lm_part, "model") and isinstance(lm_part.model, torch.nn.Module):
                            lm = lm_part.model
                        else:
                            lm = lm_part
                        break
            except Exception:
                pass
       
    if lm is not None:
        lm = lm.to(dtype=weight_dtype)

    # ================= 5. 构建 Pipeline =================
    print("Building Pipeline...")
    try:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler, subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae")
        
        clip, clip_tok = None, None
        if args.type == "fuse-dit-clip":
            clip = CLIPTextModelWithProjection.from_pretrained(args.clip_l, subfolder="text_encoder")
            clip_tok = CLIPTokenizer.from_pretrained(args.clip_l, subfolder="tokenizer")

        pipeline = None
        if args.type == "dit":
            pipeline = DiTPipeline(transformer=transformer, scheduler=scheduler, vae=vae, tokenizer=tokenizer, llm=lm)
        elif args.type == "adafusedit":
            pipeline = AdaFuseDiTPipeline(transformer=transformer, scheduler=scheduler, vae=vae, tokenizer=tokenizer, llm=lm)
        elif args.type == "fuse-dit":
            pipeline = FuseDiTPipeline(transformer=transformer, scheduler=scheduler, vae=vae, tokenizer=tokenizer)
        elif args.type == "fuse-dit-clip":
            pipeline = FuseDiTPipelineWithCLIP(transformer=transformer, scheduler=scheduler, vae=vae, tokenizer=tokenizer, clip=clip, clip_tokenizer=clip_tok)
            
    except Exception as e:
        print(f"Error building pipeline: {e}")
        return

    # ================= 6. 保存 =================
    
    # [关键修改] 自动判断文件夹名称
    # 逻辑：只要显式用了 --use_ema，或者直接加载的文件名里包含 'ema'，就加后缀
    is_ema_mode = False
    
    if args.use_ema:
        is_ema_mode = True
    elif os.path.isfile(args.checkpoint) and "ema" in os.path.basename(args.checkpoint).lower():
        is_ema_mode = True
        
    folder_name = "pipeline_ema" if is_ema_mode else "pipeline"
    
    # 确定基础路径
    if os.path.isfile(args.checkpoint):
        save_base = os.path.dirname(args.checkpoint)
    else:
        save_base = args.checkpoint

    output_dir = os.path.join(save_base, folder_name)
    
    print(f"Saving pipeline to: {output_dir}")
    pipeline.save_pretrained(output_dir)
    print("Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 可以传入目录，也可以直接传入 ema.pt 的绝对路径
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory or specific .pt file")
    
    # 选项：是否优先使用 EMA
    parser.add_argument("--use_ema", action="store_true", help="If loading from a directory, prefer 'ema.pt', and save to 'pipeline_ema'")
    
    parser.add_argument("--type", type=str, default="fuse-dit", choices=["dit", "fuse-dit", "fuse-dit-clip", "adafusedit"])
    parser.add_argument("--llm_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    
    parser.add_argument("--scheduler", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--clip_l", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    
    args = parser.parse_args()
    main(args)