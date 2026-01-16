import argparse
import os
import numpy as np


def load_weights(input_dir: str):
    weights_path = os.path.join(input_dir, "fusion_weights.npy")
    data = np.load(weights_path, allow_pickle=True).item()
    timesteps = np.array(data["timesteps"])
    weights = {int(k): np.array(v) for k, v in data["weights"].items()}  # layer_idx -> (T, L)
    layer_ids = sorted(weights.keys())
    W = np.stack([weights[k] for k in layer_ids], axis=0)  # (K, T, L)
    return timesteps, layer_ids, W


def rfft_power(x: np.ndarray, axis: int):
    """
    实数序列功率谱：返回频率和功率 |FFT|^2
    """
    n = x.shape[axis]
    spec = np.fft.rfft(x, axis=axis)
    power = (spec.real ** 2 + spec.imag ** 2) / max(n, 1)
    freqs = np.fft.rfftfreq(n, d=1.0)
    return freqs, power


def main():
    parser = argparse.ArgumentParser(description="Compute time/depth spectrum of fusion weights.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing fusion_weights.npy")
    parser.add_argument("--output_dir", type=str, default="fusion_spectrum", help="Directory to save spectra")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    timesteps, layer_ids, W = load_weights(args.input_dir)
    K, T, L = W.shape

    # 沿文本层取均值，得到每层的时间序列 (K, T)
    seq_time = W.mean(axis=2)
    freqs_t, power_time = rfft_power(seq_time, axis=1)  # power_time: (K, Ft)

    # 沿文本层取均值，转置得到每个时间步的层深序列 (T, K)
    seq_depth = seq_time.transpose(1, 0)  # (T, K)
    freqs_d, power_depth = rfft_power(seq_depth, axis=1)  # power_depth: (T, Fd)

    # 汇总统计
    avg_power_time = power_time.mean(axis=0)
    avg_power_depth = power_depth.mean(axis=0)

    np.savez(
        os.path.join(args.output_dir, "fusion_spectrum.npz"),
        freqs_time=freqs_t,
        power_time=power_time,
        freqs_depth=freqs_d,
        power_depth=power_depth,
        avg_power_time=avg_power_time,
        avg_power_depth=avg_power_depth,
        timesteps=timesteps,
        layer_ids=np.array(layer_ids),
    )

    # 打印简单摘要
    print(f"K={K}, T={T}, L={L}")
    print("Time spectrum (mean over layers):")
    print(f"  DC={avg_power_time[0]:.4f}, max_nonzero={avg_power_time[1:].max():.4f}")
    print("Depth spectrum (mean over timesteps):")
    print(f"  DC={avg_power_depth[0]:.4f}, max_nonzero={avg_power_depth[1:].max():.4f}")
    print(f"Saved spectra to {os.path.join(args.output_dir, 'fusion_spectrum.npz')}")


if __name__ == "__main__":
    main()
