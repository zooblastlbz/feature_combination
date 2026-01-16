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
        t_labels = [int(t) for t in timesteps[::-1]]
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
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing fusion_weights.npy, fusion_mean_over_text.json, fusion_layer_corr.csv")
    parser.add_argument("--output_pdf", type=str, default="fusion_trends.pdf", help="Path to combined PDF")
    args = parser.parse_args()

    weights_path = os.path.join(args.input_dir, "fusion_weights.npy")
    mean_path = os.path.join(args.input_dir, "fusion_mean_over_text.json")
    corr_path = os.path.join(args.input_dir, "fusion_layer_corr.csv")

    timesteps, weights, mean_over_text, layer_ids, corr = load_data(weights_path, mean_path, corr_path)

    # 临时输出单图
    weights_pdf = os.path.join(args.input_dir, "fusion_weights_heatmap.pdf")
    mean_pdf = os.path.join(args.input_dir, "fusion_mean_trend.pdf")
    corr_pdf = os.path.join(args.input_dir, "fusion_layer_corr.pdf")

    plot_weights(timesteps, weights, weights_pdf)
    plot_mean_trends(timesteps, mean_over_text, mean_pdf)
    plot_corr(layer_ids, corr, corr_pdf)

    # 合并为一个 PDF（matplotlib 不直接合并，这里简单串联）
    try:
        from PyPDF2 import PdfMerger

        merger = PdfMerger()
        for p in [weights_pdf, mean_pdf, corr_pdf]:
            merger.append(p)
        merger.write(args.output_pdf)
        merger.close()
        print(f"Saved combined PDF to {args.output_pdf}")
    except Exception as exc:
        print(f"[warn] Failed to merge PDFs ({exc}); individual PDFs saved in {args.input_dir}")


if __name__ == "__main__":
    main()
