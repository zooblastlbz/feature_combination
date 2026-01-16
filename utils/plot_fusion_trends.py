import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(weights_path, mean_path, corr_path):
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


def plot_weights(timesteps, weights, output_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 300})
    # 绘制每个层的权重随 timestep 变化（文本层维度在 y 轴）
    num_layers = len(weights)
    cols = min(4, num_layers) if num_layers > 1 else 1
    rows = (num_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    # 将时间轴反转（高->低），t=1...0
    for idx, (layer_idx, w) in enumerate(sorted(weights.items())):
        ax = axes[idx]
        w_plot = w[::-1]  # flip time axis
        t_norm = timesteps / (timesteps.max() if timesteps.max() > 0 else 1)
        t_labels_full = t_norm[::-1]
        t_labels = []
        for i, _ in enumerate(t_labels_full):
            if i == 0:
                t_labels.append("1")
            elif i == len(t_labels_full) - 1:
                t_labels.append("0")
            else:
                t_labels.append("")
        sns.heatmap(w_plot.T, ax=ax, cmap="viridis", cbar=True,
                    xticklabels=t_labels,
                    yticklabels=[f"L{i}" for i in range(w.shape[1])])
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Text Layer")

    for j in range(len(weights), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mean_trends(timesteps, mean_over_text, output_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 300})
    fig, ax = plt.subplots(figsize=(8, 4))
    for layer_idx, vals in sorted(mean_over_text.items()):
        ax.plot(timesteps, vals, label=f"Layer {layer_idx}", marker="o", markersize=3)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Fusion Weight")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_corr(layer_ids, corr, output_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 300})
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, xticklabels=layer_ids, yticklabels=layer_ids, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Layer Correlation (Pearson)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot fusion trends and correlations from analyze_fusion_trends outputs.")
    parser.add_argument("--input_dir", type=str, default="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/feature_combination/visual/timewise-layerwsie-weight", help="Directory containing fusion_weights.npy, fusion_mean_over_text.json, fusion_layer_corr.csv")
    parser.add_argument("--output_pdf", type=str, default="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/feature_combination/visual/fusion_trends.pdf", help="Path to combined PDF")
    args = parser.parse_args()


    weights_path = os.path.join(args.input_dir, "fusion_weights.npy")
    mean_path = os.path.join(args.input_dir, "fusion_mean_over_text.json")
    corr_path = os.path.join(args.input_dir, "fusion_layer_corr.csv")

    timesteps, weights, mean_over_text, layer_ids, corr = load_data(weights_path, mean_path, corr_path)

    # Compute JS distance matrix between time-marginal (T,L) and depth-marginal (K,L)
    def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        return 0.5 * (kl_pm + kl_qm)

    def cosine_sim(p: np.ndarray, q: np.ndarray) -> float:
        num = float(np.dot(p, q))
        den = (np.linalg.norm(p) * np.linalg.norm(q) + 1e-9)
        return num / den

    layer_ids_sorted = sorted(weights.keys())
    W = np.stack([weights[k] for k in layer_ids_sorted], axis=0)  # (K, T, L)
    avg_over_layers = W.mean(axis=0)  # (T, L)
    avg_over_time = W.mean(axis=1)    # (K, L)

    js_matrix = np.zeros((avg_over_layers.shape[0], avg_over_time.shape[0]))
    cos_matrix = np.zeros((avg_over_layers.shape[0], avg_over_time.shape[0]))
    for t in range(js_matrix.shape[0]):
        for k in range(js_matrix.shape[1]):
            js_matrix[t, k] = js_divergence(avg_over_layers[t], avg_over_time[k])
            cos_matrix[t, k] = cosine_sim(avg_over_layers[t], avg_over_time[k])

    # 时间步之间的 Pearson 相关（基于时间边际分布展平）
    time_corr = np.corrcoef(avg_over_layers.reshape(avg_over_layers.shape[0], -1))

    # Save JS matrix
    js_path = os.path.join(args.input_dir, "fusion_time_depth_js.csv")
    header_js = ",".join([str(lid) for lid in layer_ids_sorted])
    np.savetxt(js_path, js_matrix, delimiter=",", header=header_js, comments="", fmt="%.6f")
    # Save cosine matrix
    cos_path = os.path.join(args.input_dir, "fusion_time_depth_cosine.csv")
    np.savetxt(cos_path, cos_matrix, delimiter=",", header=header_js, comments="", fmt="%.6f")
    # Save time-time Pearson corr
    time_corr_path = os.path.join(args.input_dir, "fusion_time_corr.csv")
    np.savetxt(time_corr_path, time_corr, delimiter=",", comments="", fmt="%.6f")

    # 临时输出单图
    weights_pdf = os.path.join(args.input_dir, "fusion_weights_heatmap.pdf")
    mean_pdf = os.path.join(args.input_dir, "fusion_mean_trend.pdf")
    corr_pdf = os.path.join(args.input_dir, "fusion_layer_corr.pdf")
    js_pdf = os.path.join(args.input_dir, "fusion_time_depth_js.pdf")
    cos_pdf = os.path.join(args.input_dir, "fusion_time_depth_cosine.pdf")
    time_corr_pdf = os.path.join(args.input_dir, "fusion_time_corr.pdf")

    plot_weights(timesteps, weights, weights_pdf)
    plot_mean_trends(timesteps, mean_over_text, mean_pdf)
    plot_corr(layer_ids, corr, corr_pdf)
    # JS heatmap
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 300})
    fig, ax = plt.subplots(figsize=(6, 5))
    t_norm = timesteps / (timesteps.max() if timesteps.max() > 0 else 1)
    t_labels = []
    for i in range(len(t_norm)):
        if i == 0:
            t_labels.append("0")
        elif i == len(t_norm) - 1:
            t_labels.append("1")
        else:
            t_labels.append("")
    sns.heatmap(js_matrix, xticklabels=layer_ids_sorted, yticklabels=t_labels[::-1], cmap="magma", ax=ax)
    ax.set_xlabel("DiT Layer")
    ax.set_ylabel("Timestep (reversed)")
    ax.set_title("JS distance (time marginal vs depth marginal)")
    fig.tight_layout()
    fig.savefig(js_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Cosine heatmap
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 300})
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cos_matrix, xticklabels=layer_ids_sorted, yticklabels=t_labels[::-1], cmap="viridis", ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("DiT Layer")
    ax.set_ylabel("Timestep (reversed)")
    ax.set_title("Cosine similarity (time marginal vs depth marginal)")
    fig.tight_layout()
    fig.savefig(cos_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Time-time Pearson correlation heatmap
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 300})
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(time_corr, xticklabels=t_labels, yticklabels=t_labels, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_xlabel("Timestep (normalized)")
    ax.set_ylabel("Timestep (normalized)")
    ax.set_title("Pearson correlation between timesteps")
    fig.tight_layout()
    fig.savefig(time_corr_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 合并为一个 PDF（matplotlib 不直接合并，这里简单串联）
    try:
        from PyPDF2 import PdfMerger

        merger = PdfMerger()
        for p in [weights_pdf, mean_pdf, corr_pdf, js_pdf, cos_pdf, time_corr_pdf]:
            merger.append(p)
        merger.write(args.output_pdf)
        merger.close()
        print(f"Saved combined PDF to {args.output_pdf}")
    except Exception as exc:
        print(f"[warn] Failed to merge PDFs ({exc}); individual PDFs saved in {args.input_dir}")


if __name__ == "__main__":
    main()
