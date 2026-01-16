import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# allow importing diffusion
sys.path.append(os.getcwd())

from diffusion.models import build_model
from diffusion.pipelines import AdaFuseDiTPipeline


def _parse_layer_list(layer_str: str):
    if not layer_str:
        return None
    layers = []
    for part in layer_str.split(","):
        part = part.strip()
        if part == "":
            continue
        layers.append(int(part))
    return set(layers)


def load_checkpoint(model, checkpoint_path: str):
    state_dict = None
    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    elif os.path.isdir(checkpoint_path):
        p = os.path.join(checkpoint_path, "mp_rank_00_model_states.pt")
        if os.path.exists(p):
            payload = torch.load(p, map_location="cpu")
            state_dict = payload["module"] if "module" in payload else payload
    if state_dict is None:
        raise ValueError(f"Could not load state dict from {checkpoint_path}")
    clean = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean, strict=False)


def is_pipeline_dir(path: str):
    if not os.path.isdir(path):
        return False
    indicators = ["model_index.json", "config.json", "pytorch_model.bin", "model.safetensors"]
    return any(os.path.exists(os.path.join(path, f)) for f in indicators)


def load_model_from_args(args):
    config = OmegaConf.load(args.config)

    class HParams:
        def __init__(self, cfg):
            self.model = cfg.model if "model" in cfg else cfg
            self.trainer = argparse.Namespace(train_dit=False, train_llm=False)

    hparams = HParams(config)

    if is_pipeline_dir(args.checkpoint):
        pipe = AdaFuseDiTPipeline.from_pretrained(args.checkpoint)
        return pipe.transformer

    model = build_model(hparams)
    load_checkpoint(model, args.checkpoint)
    return model


def extract_weights(model, num_timesteps: int, layer_filter=None):
    """
    按 visualize_fusion_weights 的方式获取 timewise+layerwise 融合权重。
    返回 dict: layer_idx -> ndarray (T, num_text_layers)
    """
    assert model.config.use_layer_wise_fusion, "Expect use_layer_wise_fusion=True"
    assert model.config.use_timestep_adaptive_fusion, "Expect use_timestep_adaptive_fusion=True"

    timesteps = torch.linspace(0, 1000, num_timesteps).long()
    t_input = timesteps.float() / 1000.0
    t_input = t_input.to(model.device)

    results: Dict[int, np.ndarray] = {}
    for i, module in enumerate(model.text_fusion_modules):
        if layer_filter is not None and i not in layer_filter:
            continue
        with torch.no_grad():
            dummy_hidden = [
                torch.zeros(1, 1, module.feature_dim, device=t_input.device, dtype=t_input.dtype)
                for _ in range(module.num_layers)
            ]
            _, w = module(dummy_hidden, t_input)  # (T, num_text_layers)
            results[i] = w.cpu().numpy()
    return timesteps.cpu().numpy(), results


def compute_trends(timesteps: np.ndarray, weights: Dict[int, np.ndarray]):
    """
    weights: layer_idx -> (T, L)
    返回:
      - mean_over_text: layer_idx -> (T,) 每个 timestep 的权重均值
      - corr_matrix: KxK Pearson，相对整个权重曲线 (flatten over T,L)
    """
    mean_over_text = {k: v.mean(axis=1) for k, v in weights.items()}

    layer_ids: List[int] = sorted(weights.keys())
    flat = [weights[k].reshape(-1) for k in layer_ids]
    mat = np.stack(flat, axis=0)
    # Pearson 相关
    mat_norm = (mat - mat.mean(axis=1, keepdims=True)) / (mat.std(axis=1, keepdims=True) + 1e-8)
    corr = mat_norm @ mat_norm.T / mat.shape[1]
    return mean_over_text, layer_ids, corr


def save_outputs(args, timesteps, weights, mean_over_text, layer_ids, corr):
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存原始权重
    weight_path = os.path.join(args.output_dir, "fusion_weights.npy")
    np.save(weight_path, {"timesteps": timesteps, "weights": weights})

    # 保存 mean_over_text
    mean_path = os.path.join(args.output_dir, "fusion_mean_over_text.json")
    payload = {
        "timesteps": timesteps.tolist(),
        "mean_over_text": {int(k): v.tolist() for k, v in mean_over_text.items()},
    }
    with open(mean_path, "w") as f:
        json.dump(payload, f, indent=2)

    # 保存相关矩阵
    corr_path = os.path.join(args.output_dir, "fusion_layer_corr.csv")
    header = ",".join([str(i) for i in layer_ids])
    np.savetxt(corr_path, corr, delimiter=",", header=header, comments="", fmt="%.6f")

    print(f"Saved weights to {weight_path}")
    print(f"Saved mean_over_text to {mean_path}")
    print(f"Saved layer correlation to {corr_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze timewise+layerwise fusion weights trends and correlations.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path (pipeline dir or model weights).")
    parser.add_argument("--config", type=str, required=True, help="Config yaml for model (ignored if pipeline dir).")
    parser.add_argument("--num_timesteps", type=int, default=20, help="Number of timesteps to sample between 0 and 1000.")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated DiT layer ids to include.")
    parser.add_argument("--output_dir", type=str, default="fusion_analysis", help="Directory to save results.")
    args = parser.parse_args()

    model = load_model_from_args(args)
    model.eval()

    layer_filter = _parse_layer_list(args.layers) if args.layers else None

    timesteps, weights = extract_weights(model, args.num_timesteps, layer_filter)
    if not weights:
        print("No fusion weights extracted; check layer list or model config.")
        return

    mean_over_text, layer_ids, corr = compute_trends(timesteps, weights)
    save_outputs(args, timesteps, weights, mean_over_text, layer_ids, corr)


if __name__ == "__main__":
    main()
