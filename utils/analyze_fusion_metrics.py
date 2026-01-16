import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from diffusion.models import build_model
from diffusion.pipelines import AdaFuseDiTPipeline


def load_model(checkpoint: str, config: str):
    def is_pipeline_dir(path: str):
        if not os.path.isdir(path):
            return False
        indicators = ["model_index.json", "config.json", "pytorch_model.bin", "model.safetensors"]
        return any(os.path.exists(os.path.join(path, f)) for f in indicators)

    if is_pipeline_dir(checkpoint):
        pipe = AdaFuseDiTPipeline.from_pretrained(checkpoint)
        return pipe.transformer

    cfg = OmegaConf.load(config)

    class HParams:
        def __init__(self, cfg):
            self.model = cfg.model if "model" in cfg else cfg
            self.trainer = argparse.Namespace(train_dit=False, train_llm=False)

    hparams = HParams(cfg)
    model = build_model(hparams)

    state = torch.load(checkpoint, map_location="cpu")
    state = state["module"] if isinstance(state, dict) and "module" in state else state
    clean = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(clean, strict=False)
    return model


@torch.no_grad()
def extract_weights(transformer, num_timesteps: int):
    timesteps = torch.linspace(0, 1000, num_timesteps).long()
    t_input = timesteps.float() / 1000.0
    t_input = t_input.to(transformer.device)

    if hasattr(transformer, "text_fusion_modules") and transformer.text_fusion_modules is not None:
        weights = {}
        for i, module in enumerate(transformer.text_fusion_modules):
            dummy_hidden = [
                torch.zeros(1, 1, module.feature_dim, device=t_input.device, dtype=t_input.dtype)
                for _ in range(module.num_layers)
            ]
            _, w = module(dummy_hidden, t_input)  # (T, L)
            weights[i] = w.cpu().numpy()
        mode = "layer_timewise" if transformer.config.use_timestep_adaptive_fusion else "layer_static"
    elif hasattr(transformer, "text_fusion_module") and transformer.text_fusion_module is not None:
        module = transformer.text_fusion_module
        dummy_hidden = [
            torch.zeros(1, 1, module.feature_dim, device=t_input.device, dtype=t_input.dtype)
            for _ in range(module.num_layers)
        ]
        _, w = module(dummy_hidden, t_input)  # (T, L)
        weights = {"global": w.cpu().numpy()}
        mode = "global_timewise"
    elif hasattr(transformer, "text_fusion_weights") and transformer.text_fusion_weights is not None:
        weights = {}
        for i, param in enumerate(transformer.text_fusion_weights):
            w = torch.softmax(param, dim=0).unsqueeze(0).repeat(len(timesteps), 1)
            weights[i] = w.detach().cpu().numpy()
        mode = "layer_static"
    elif hasattr(transformer, "text_fusion_weight") and transformer.text_fusion_weight is not None:
        w = torch.softmax(transformer.text_fusion_weight, dim=0).unsqueeze(0).repeat(len(timesteps), 1)
        weights = {"global": w.detach().cpu().numpy()}
        mode = "global_static"
    else:
        raise ValueError("No fusion modules found.")

    return timesteps.cpu().numpy(), weights, mode


def compute_metrics(timesteps, weights_dict):
    metrics = {}
    for key, w in weights_dict.items():
        # w: (T, L)
        entropy = -np.sum(w * np.log(np.clip(w, 1e-9, 1.0)), axis=1)
        maxv = w.max(axis=1)
        argmax = w.argmax(axis=1)
        metrics[key] = {
            "entropy_mean": float(entropy.mean()),
            "entropy_std": float(entropy.std()),
            "max_mean": float(maxv.mean()),
            "max_std": float(maxv.std()),
            "top_idx_hist": {int(i): int((argmax == i).sum()) for i in range(w.shape[1])},
        }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute fusion weight metrics for different fusion modes.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True, help="Config yaml (ignored if checkpoint is pipeline dir)")
    parser.add_argument("--num_timesteps", type=int, default=20)
    parser.add_argument("--output_json", type=str, default="fusion_metrics.json")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.config)
    model.eval()

    timesteps, weights, mode = extract_weights(model, args.num_timesteps)
    metrics = compute_metrics(timesteps, weights)

    payload = {
        "mode": mode,
        "timesteps": timesteps.tolist(),
        "metrics": metrics,
    }
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics to {args.output_json} (mode={mode})")


if __name__ == "__main__":
    main()
