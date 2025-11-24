import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

# Add current directory to sys.path to allow importing diffusion
sys.path.append(os.getcwd())

from diffusion.models import build_model

def load_checkpoint(model, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = None
    
    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    elif os.path.isdir(checkpoint_path):
        # Try to find mp_rank_00_model_states.pt (common in DeepSpeed)
        p = os.path.join(checkpoint_path, "mp_rank_00_model_states.pt")
        if os.path.exists(p):
            print(f"Found model states at {p}")
            payload = torch.load(p, map_location="cpu")
            if "module" in payload:
                state_dict = payload["module"]
            else:
                state_dict = payload
        else:
            # Try DeepSpeed ZeRO utility
            print("Attempting to load using DeepSpeed ZeRO utility...")
            try:
                from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
                state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
            except ImportError:
                print("DeepSpeed not installed or failed to import. Cannot load ZeRO checkpoint.")
            except Exception as e:
                print(f"Failed to load ZeRO checkpoint: {e}")
                
    if state_dict is None:
        raise ValueError(f"Could not load state dict from {checkpoint_path}")

    # Handle 'module.' prefix if present (DDP/DeepSpeed)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Checkpoint loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

def visualize(args):
    # Load config
    print(f"Loading config from {args.config}...")
    config = OmegaConf.load(args.config)
    
    # Mock HParams
    class HParams:
        def __init__(self, config):
            self.model = config.model if "model" in config else config
            self.trainer = argparse.Namespace(train_dit=False, train_llm=False)
            
    hparams = HParams(config)
    
    # Build model
    print("Building model...")
    # We can try to avoid loading the heavy LLM if we only need DiT weights,
    # but build_model is coupled. We'll just load it.
    model = build_model(hparams)
    model.eval()
    
    # Load weights
    load_checkpoint(model, args.checkpoint)
    
    # Prepare timesteps
    # Generate timesteps from 0 to 1000
    timesteps = torch.linspace(0, 1000, args.num_timesteps).long()
    # AdaFuseDiT uses normalized timesteps [0, 1] for the fusion module
    t_input = timesteps.float() / 1000.0
    t_input = t_input.to(model.device) # Model might be on CPU, which is fine
    
    results = {} # key -> weights (T, NumTextLayers)
    
    print("Extracting fusion weights...")
    if hasattr(model, "text_fusion_modules") and model.text_fusion_modules is not None:
        print("Detected Layer-wise Adaptive Fusion")
        for i, module in enumerate(model.text_fusion_modules):
            with torch.no_grad():
                t_embed = module._time_embedding(t_input)
                w = module.weight_generator(t_embed) # (T, NumTextLayers)
                results[f"Layer_{i}"] = w.cpu().numpy()
                
    elif hasattr(model, "text_fusion_module") and model.text_fusion_module is not None:
        print("Detected Global Adaptive Fusion")
        module = model.text_fusion_module
        with torch.no_grad():
            t_embed = module._time_embedding(t_input)
            w = module.weight_generator(t_embed)
            results["Global"] = w.cpu().numpy()
            
    elif hasattr(model, "text_fusion_weights") and model.text_fusion_weights is not None:
        print("Detected Layer-wise Static Fusion")
        for i, param in enumerate(model.text_fusion_weights):
            w = torch.softmax(param, dim=0).unsqueeze(0).repeat(len(timesteps), 1)
            results[f"Layer_{i}"] = w.detach().cpu().numpy()
            
    elif hasattr(model, "text_fusion_weight") and model.text_fusion_weight is not None:
        print("Detected Global Static Fusion")
        w = torch.softmax(model.text_fusion_weight, dim=0).unsqueeze(0).repeat(len(timesteps), 1)
        results["Global"] = w.detach().cpu().numpy()
        
    else:
        print("No fusion modules found in model. Is this AdaFuseDiT?")
        return

    # Plotting
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use a colormap
    cm = plt.get_cmap("viridis")
    
    for key, weights in results.items():
        # Save to CSV
        num_text_layers = weights.shape[1]
        csv_filename = f"fusion_weights_{key}.csv"
        csv_path = os.path.join(args.output_dir, csv_filename)
        
        # Prepare header
        header = "Timestep," + ",".join([f"TextLayer_{i}" for i in range(num_text_layers)])
        
        # Prepare data with timestep column
        t_np = timesteps.cpu().numpy()
        data_to_save = np.column_stack((t_np, weights))
        
        np.savetxt(csv_path, data_to_save, delimiter=",", header=header, comments="", fmt="%.6f")
        print(f"Saved weights to {csv_path}")

        plt.figure(figsize=(30, 18))
        # weights: (T, N)
        num_text_layers = weights.shape[1]
        x = np.arange(num_text_layers)
        
        for t_idx, t_val in enumerate(timesteps):
            color = cm(t_idx / (len(timesteps) - 1) if len(timesteps) > 1 else 0.5)
            plt.plot(x, weights[t_idx], label=f"t={int(t_val)}", color=color, marker='o', markersize=4)
            
        plt.xlabel("Text Encoder Layer Index")
        plt.ylabel("Fusion Weight (Softmax)")
        plt.title(f"Fusion Weights - {key}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(x) # Ensure integer ticks for layers
        plt.tight_layout()
        
        filename = f"fusion_weights_{key}.pdf"
        save_path = os.path.join(args.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize AdaFuseDiT fusion weights")
    parser.add_argument("--checkpoint", type=str, default="/ytech_m2v5_hdd/workspace/kling_mm/libozhou/feature_combination/output/256-AdaFuseDiT-timewise/115000", help="Path to the checkpoint (file or folder)")
    parser.add_argument("--config", type=str, default="/ytech_m2v5_hdd/workspace/kling_mm/libozhou/feature_combination/configs/adafusedit/qwen3-vl-4b-4machine.yaml", help="Path to the model config file (.yaml)")
    parser.add_argument("--output_dir", type=str, default="visual/fusion_plots_115000", help="Directory to save plots")
    parser.add_argument("--num_timesteps", type=int, default=10, help="Number of timestep curves to plot")
    
    args = parser.parse_args()
    visualize(args)
