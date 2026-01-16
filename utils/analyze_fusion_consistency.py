import argparse
import json
import os
import numpy as np


def load_outputs(input_dir: str):
    weights_path = os.path.join(input_dir, "fusion_weights.npy")
    mean_path = os.path.join(input_dir, "fusion_mean_over_text.json")
    corr_path = os.path.join(input_dir, "fusion_layer_corr.csv")

    data = np.load(weights_path, allow_pickle=True).item()
    timesteps = np.array(data["timesteps"])
    weights = {int(k): np.array(v) for k, v in data["weights"].items()}

    with open(mean_path, "r") as f:
        mean_data = json.load(f)
    mean_over_text = {int(k): np.array(v) for k, v in mean_data["mean_over_text"].items()}

    corr = np.loadtxt(corr_path, delimiter=",", skiprows=1)
    with open(corr_path, "r") as f:
        header = f.readline().strip().lstrip("#").split(",")
    layer_ids = [int(h) for h in header]
    return timesteps, weights, mean_over_text, layer_ids, corr


def analyze_consistency(timesteps, mean_over_text, layer_ids):
    """
    mean_over_text: dict layer_idx -> (T,) 已对文本层做均值的权重曲线
    返回:
      - global_corr: flatten 的时间变化与层深变化之间的 Pearson
      - per_layer_trend: 每层随时间的 Spearman rho
      - per_timestep_trend: 每个时间点随层深的 Spearman rho
    """
    # 构造矩阵 M: (K, T)
    K = len(layer_ids)
    T = len(timesteps)
    M = np.zeros((K, T), dtype=np.float32)
    for i, lid in enumerate(layer_ids):
        M[i] = mean_over_text[lid]

    # 时间方向与层深方向的增量
    d_time = np.diff(M, axis=1)  # (K, T-1)
    d_depth = np.diff(M, axis=0)  # (K-1, T)

    # 全局相关：flatten 两个增量矩阵
    dt_flat = d_time.flatten()
    dd_flat = d_depth.flatten()
    dt_norm = (dt_flat - dt_flat.mean()) / (dt_flat.std() + 1e-8)
    dd_norm = (dd_flat - dd_flat.mean()) / (dd_flat.std() + 1e-8)
    global_corr = float(np.dot(dt_norm, dd_norm) / len(dt_norm))

    # 每层随时间的 Spearman rho
    per_layer_rho = []
    for i in range(K):
        rho = np.corrcoef(M[i], np.argsort(timesteps))[0, 1] if T > 1 else 0.0
        per_layer_rho.append((layer_ids[i], float(rho)))

    # 每个时间点随层深的 Spearman rho
    per_t_rho = []
    depth_index = np.arange(K)
    for t in range(T):
        rho = np.corrcoef(M[:, t], depth_index)[0, 1] if K > 1 else 0.0
        per_t_rho.append((int(timesteps[t]), float(rho)))

    return global_corr, per_layer_rho, per_t_rho


def main():
    parser = argparse.ArgumentParser(description="Check consistency between timestep and layer-depth trends.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing fusion_weights.npy etc.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to dump JSON results.")
    args = parser.parse_args()

    timesteps, weights, mean_over_text, layer_ids, corr = load_outputs(args.input_dir)

    global_corr, per_layer_rho, per_t_rho = analyze_consistency(timesteps, mean_over_text, layer_ids)

    print(f"Global correlation between time-variation and depth-variation (delta-based): {global_corr:.4f}")
    print("Per-layer Spearman (weights vs timestep order):")
    for lid, rho in per_layer_rho:
        print(f"  Layer {lid}: rho={rho:.4f}")
    print("Per-timestep Spearman (weights vs layer depth):")
    for t, rho in per_t_rho:
        print(f"  t={t}: rho={rho:.4f}")

    if args.output_json:
        payload = {
            "global_corr": global_corr,
            "per_layer_spearman": per_layer_rho,
            "per_timestep_spearman": per_t_rho,
            "layer_ids": layer_ids,
            "timesteps": timesteps.tolist(),
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
