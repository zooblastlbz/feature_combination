import argparse
import io
import os
import re
from typing import Optional, Tuple, Dict

import torch
import zstandard as zstd
from omegaconf import OmegaConf
import csv

# --- 确保可从任意位置运行脚本 ---
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from diffusion.models import build_model


def _find_latest_step(ckpt_dir: str) -> Optional[int]:
    latest_file = os.path.join(ckpt_dir, "latest")
    if os.path.exists(latest_file):
        try:
            with open(latest_file) as f:
                return int(f.read().strip())
        except Exception:
            pass
    # 回退：扫描数字子目录
    steps = []
    for name in os.listdir(ckpt_dir):
        if os.path.isdir(os.path.join(ckpt_dir, name)) and re.fullmatch(r"\d+", name):
            steps.append(int(name))
    return max(steps) if steps else None


def _load_consolidated_state(step_dir: str, use_ema: bool) -> Optional[Dict[str, torch.Tensor]]:
    fname = "ema.pt.zst" if use_ema else "model.pt.zst"
    fpath = os.path.join(step_dir, fname)
    if not os.path.exists(fpath):
        return None
    with open(fpath, "rb") as f:
        data = zstd.decompress(f.read())
    buffer = io.BytesIO(data)
    state = torch.load(buffer, map_location="cpu")
    return state


def _load_from_zero_ckpt(ckpt_dir: str, tag: str) -> Optional[Dict[str, torch.Tensor]]:
    try:
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    except Exception:
        return None
    try:
        state = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir, tag=tag)
        return state
    except Exception:
        return None


def load_model_weights(model: torch.nn.Module, ckpt_dir: str, step: Optional[int], use_ema: bool) -> Tuple[torch.nn.Module, int]:
    """
    返回已加载权重的模型与实际使用的 step。
    """
    if step is None:
        step = _find_latest_step(ckpt_dir)
        if step is None:
            raise FileNotFoundError(f"未在 {ckpt_dir} 找到任何 checkpoint 目录")

    step_dir = os.path.join(ckpt_dir, str(step))

    # 1) 优先使用已整合的 zst 权重
    state = _load_consolidated_state(step_dir, use_ema)

    # 2) 若不存在，尝试从 DeepSpeed Zero 分片提取
    if state is None:
        state = _load_from_zero_ckpt(ckpt_dir, tag=str(step))

    if state is None:
        raise FileNotFoundError(
            f"未找到可用的权重文件：{step_dir}/(ema.pt.zst|model.pt.zst)，且无法从 DeepSpeed 分片提取"
        )

    # 兼容 'module.' 前缀
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[警告] 有缺失参数未加载：{len(missing)} 个（与绘图无关可忽略）")
    if unexpected:
        print(f"[警告] 有多余参数：{len(unexpected)} 个（与绘图无关可忽略）")

    model.eval()
    return model, step


def compute_weights_over_timesteps(model: torch.nn.Module, t_min: int, t_max: int, num_points: int, per_layer: bool):
    """
    返回用于绘图的数据结构：
    - 如果 per_layer 且存在 text_fusion_modules：
      dict[layer_idx -> {"t": list[int], "w": Tensor[num_points, K]}]
    - 否则：
      {"t": list[int], "w": Tensor[num_points, K]}
    """
    # 兼容不同模式
    fusion_modules = None
    if hasattr(model, "text_fusion_modules"):
        fusion_modules = list(model.text_fusion_modules)
    elif hasattr(model, "text_fusion_module"):
        fusion_modules = [model.text_fusion_module]
    else:
        # 非时间自适应模式，只有固定可学习权重
        if hasattr(model, "text_fusion_weights"):
            # per-layer 固定权重
            data = {}
            for li, w in enumerate(model.text_fusion_weights):
                weights = torch.softmax(w.detach().cpu(), dim=0).unsqueeze(0).repeat(num_points, 1)
                t_values = [int(t) for t in torch.linspace(t_min, t_max, steps=num_points).tolist()]
                data[li] = {"t": t_values, "w": weights}
            return data
        elif hasattr(model, "text_fusion_weight"):
            # 全局固定权重
            w = model.text_fusion_weight
            weights = torch.softmax(w.detach().cpu(), dim=0).unsqueeze(0).repeat(num_points, 1)
            t_values = [int(t) for t in torch.linspace(t_min, t_max, steps=num_points).tolist()]
            return {"t": t_values, "w": weights}
        else:
            raise ValueError("当前模型不包含可绘制的融合权重模块")

    # 构造虚拟输入（仅用于通过 forward 获得权重；融合权重只依赖时间，不依赖特征值）
    K = fusion_modules[0].num_layers
    feat_dim = fusion_modules[0].feature_dim
    dummy_features = [torch.zeros(1, 1, feat_dim) for _ in range(K)]

    # 采样时间步
    t_values = torch.linspace(t_min, t_max, steps=num_points)

    if per_layer and len(fusion_modules) > 1:
        out = {}
        for li, fm in enumerate(fusion_modules):
            ws = []
            for t in t_values:
                t_norm = torch.tensor([float(t.item()) / 1000.0])
                with torch.no_grad():
                    _, w = fm(dummy_features, t_norm)
                ws.append(w.squeeze(0).cpu())
            out[li] = {"t": [int(x.item()) for x in t_values], "w": torch.stack(ws, dim=0)}
        return out
    else:
        fm = fusion_modules[0]
        ws = []
        for t in t_values:
            t_norm = torch.tensor([float(t.item()) / 1000.0])
            with torch.no_grad():
                _, w = fm(dummy_features, t_norm)
            ws.append(w.squeeze(0).cpu())
        return {"t": [int(x.item()) for x in t_values], "w": torch.stack(ws, dim=0)}


def plot_weights(data, save_path: Optional[str] = None, title: Optional[str] = None):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("未安装 matplotlib，无法绘图。请先安装：pip install matplotlib")
        return

    def _plot_one(ax, t_list, w_tensor, title_suffix: str = ""):
        # w_tensor: [N_timestep, K_layers]
        W = w_tensor.numpy()
        N, K = W.shape
        x = list(range(K))
        for i in range(N):
            ax.plot(x, W[i, :], label=f"t={t_list[i]}")
        ax.set_xlabel("layer index")
        ax.set_ylabel("weight")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
        if title_suffix:
            ax.set_title(title_suffix)
        ax.legend(ncol=2, fontsize=8)

    if isinstance(data, dict) and all(isinstance(k, int) for k in data.keys()):
        # per-layer：多子图（每个子图仍是“横轴层数、纵轴权重、每条线一个时间步”）
        L = len(data)
        import math
        cols = 3
        rows = math.ceil(L / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 3.5*rows), squeeze=False)
        for i, (li, pack) in enumerate(sorted(data.items(), key=lambda x: x[0])):
            r, c = divmod(i, cols)
            _plot_one(axes[r][c], pack["t"], pack["w"], f"Layer {li}")
        # 清理空轴
        for j in range(i+1, rows*cols):
            r, c = divmod(j, cols)
            fig.delaxes(axes[r][c])
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
            print(f"已保存图像到 {save_path}")
        else:
            plt.show()
    else:
        # 单图
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_one(ax, data["t"], data["w"], title or "")
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
            print(f"已保存图像到 {save_path}")
        else:
            plt.show()


def write_csv(data, csv_path: str):
    os.makedirs(csv_path, exist_ok=True) if os.path.isdir(csv_path) else None

    def _ensure_csv_path(path: str) -> str:
        base, ext = os.path.splitext(path)
        if ext.lower() != ".csv":
            path = path + (".csv" if not ext else "")
        return path

    def _write_one(path: str, t_list, w_tensor):
        path = _ensure_csv_path(path)
        K = w_tensor.shape[1]
        headers = ["timestep"] + [f"w{k}" for k in range(K)]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for i, t in enumerate(t_list):
                row = [int(t)] + [float(x) for x in w_tensor[i].tolist()]
                writer.writerow(row)
        print(f"已保存CSV到 {path}")

    if isinstance(data, dict) and all(isinstance(k, int) for k in data.keys()):
        # 分层：多文件
        if os.path.isdir(csv_path):
            for li, pack in data.items():
                _write_one(os.path.join(csv_path, f"weights_layer{li}.csv"), pack["t"], pack["w"])
        else:
            base, ext = os.path.splitext(csv_path)
            if not ext:
                ext = ".csv"
            for li, pack in data.items():
                _write_one(f"{base}_layer{li}{ext}", pack["t"], pack["w"])
    else:
        # 全局：单文件或目录/weights.csv
        if os.path.isdir(csv_path):
            _write_one(os.path.join(csv_path, "weights.csv"), data["t"], data["w"])
        else:
            _write_one(csv_path, data["t"], data["w"])


def main():
    parser = argparse.ArgumentParser(description="Plot AdaFuseDiT fusion weights over timesteps")

    parser.add_argument("--config", default="/ytech_m2v5_hdd/workspace/kling_mm/libozhou/feature_combination/configs/adafusedit/baseline.yaml", help="YAML 配置路径")
    parser.add_argument("--ckpt_dir", default="/ytech_m2v5_hdd/workspace/kling_mm/libozhou/output/512-AdaFuseDiT-timewise", help="checkpoint 目录（包含 step 子目录与 latest 文件）")
    parser.add_argument("--step", type=int, default=None, help="指定 step；缺省则自动寻找 latest")
    parser.add_argument("--ema", action="store_true", help="优先使用 EMA 权重（若存在）")
    parser.add_argument("--t_min", type=int, default=0)
    parser.add_argument("--t_max", type=int, default=1000)
    parser.add_argument("--points", type=int, default=51, help="采样点数")
    parser.add_argument("--per_layer", action="store_true", help="若为层级融合，按层分别绘图")
    parser.add_argument("--save", type=str, default=None, help="输出图片路径（兼容参数，支持 .png/.svg 等）")
    parser.add_argument("--save_svg", type=str, default="visual/layer_weights.svg", help="输出 SVG 路径（优先于 --save）")
    parser.add_argument("--csv", type=str, default="visual/layer_weights.csv", help="将数值保存为 CSV。全局：直接指定 .csv 或目录；分层：指定目录，或指定文件前缀（将自动生成 *_layerX.csv）")
    args = parser.parse_args()
    

    hparams = OmegaConf.load(args.config)
    model = build_model(hparams)
    model, used_step = load_model_weights(model, args.ckpt_dir, args.step, args.ema)

    data = compute_weights_over_timesteps(model, args.t_min, args.t_max, args.points, args.per_layer)

    title = f"Fusion Weights @ step {used_step}"
    if hasattr(model.config, "use_layer_wise_fusion") and model.config.use_layer_wise_fusion:
        title += " (per-layer)" if args.per_layer else " (shared-view)"
    else:
        title += " (global)"

    save_path = args.save_svg or args.save
    plot_weights(data, save_path, title)

    if args.csv:
        write_csv(data, args.csv)


if __name__ == "__main__":
    main()
