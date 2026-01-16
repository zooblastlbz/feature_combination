import argparse
import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from diffusion.pipelines import AdaFuseDiTPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze timewise + layer-wise fusion weights over timesteps and layers.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to AdaFuseDiT pipeline checkpoint.")
    parser.add_argument("--prompt", type=str, default="a colorful mural on a brick wall", help="Prompt text.")
    parser.add_argument("--resolution", type=int, default=512, help="Output resolution (height=width).")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps.")
    parser.add_argument("--guidance", type=float, default=7.0, help="CFG guidance scale.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Compute dtype.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON path to dump weight stats.")
    return parser.parse_args()


def get_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch_dtype = get_dtype(args.dtype)

    pipe = AdaFuseDiTPipeline.from_pretrained(args.checkpoint, torch_dtype=torch_dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    # 日志结构：按 layer_idx 保存不同 timestep 的权重均值
    weight_log = defaultdict(list)

    transformer = pipe.transformer

    # 仅支持 timewise + layer-wise 模式的采样
    assert transformer.config.use_layer_wise_fusion, "This script expects use_layer_wise_fusion=True"
    assert transformer.config.use_timestep_adaptive_fusion, "This script expects use_timestep_adaptive_fusion=True"

    # 记录原始方法，便于复原
    original_fuse = transformer._fuse_text_features

    def fused_with_log(text_hidden_states, timestep):
        # 基于当前实现的等价逻辑，外加权重记录（按 batch 均值）
        if text_hidden_states is None:
            return None, None

        if isinstance(text_hidden_states, (list, tuple)):
            stacked = torch.stack(text_hidden_states, dim=0).float()  # (L, B, S, C)
            text_dtype = text_hidden_states[0].dtype
        else:
            stacked = text_hidden_states.unsqueeze(0).float()
            text_dtype = text_hidden_states.dtype

        stacked = F.layer_norm(stacked, (stacked.shape[-1],))

        fused_text_per_layer = []
        normalized_timestep = timestep.float() / 1000.0
        # 共享一次时间嵌入
        t_embed = transformer.text_fusion_modules[0]._time_embedding(normalized_timestep)
        weight_logits = torch.stack(
            [module.weight_generator(t_embed.to(dtype=stacked.dtype)) for module in transformer.text_fusion_modules],
            dim=0,
        )  # (K, B, L)
        weights = F.softmax(weight_logits, dim=-1)  # (K, B, L)
        fused_stack = torch.einsum("kbl,lbsc->kbsc", weights, stacked)  # (K, B, S, C)

        # 记录权重（对 batch 求均值，便于后续分析）
        with torch.no_grad():
            t_scalar = timestep.detach().cpu().item() if timestep.numel() == 1 else timestep.detach().cpu().tolist()
            mean_weights = weights.detach().mean(dim=1).cpu()  # (K, L)
            for layer_idx in range(transformer.config.dit_num_hidden_layers):
                weight_log[layer_idx].append({"t": t_scalar, "weights": mean_weights[layer_idx].tolist()})

        for layer_idx in range(transformer.config.dit_num_hidden_layers):
            fused_layer = fused_stack[layer_idx].to(dtype=text_dtype)
            fused_text_per_layer.append(transformer.context_embedder(fused_layer).to(fused_layer.dtype))

        return None, fused_text_per_layer

    # 替换融合函数
    transformer._fuse_text_features = fused_with_log  # type: ignore

    # 执行一次推理（不关心输出，只为跑过各个 timestep）
    _ = pipe(
        prompt=args.prompt,
        height=args.resolution,
        width=args.resolution,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        output_type="latent",
    )

    # 还原原方法，避免污染
    transformer._fuse_text_features = original_fuse  # type: ignore

    # 输出统计
    print(f"Collected weights for {len(weight_log)} layers.")
    for layer_idx, entries in weight_log.items():
        ts = [e["t"] for e in entries]
        print(f"Layer {layer_idx}: {len(entries)} timesteps, t range [{min(ts)}, {max(ts)}]")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(weight_log, f, indent=2)
        print(f"Saved weight log to {args.output}")


if __name__ == "__main__":
    main()
