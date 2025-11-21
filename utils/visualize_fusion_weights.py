#!/usr/bin/env python3
"""
可视化 AdaFuseDiT 模型在不同时间步各层的融合权重

用法:
    python utils/visualize_fusion_weights.py \
        --checkpoint_path /path/to/checkpoint \
        --output_path fusion_weights.png \
        --timesteps 0 100 200 500 800 999 \
        --num_layers 18
"""

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from tqdm import tqdm

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion.models import AdaFuseDiT, TimestepWiseFeatureWeighting


def load_checkpoint(checkpoint_path):
    """加载 checkpoint (支持 Accelerate/DeepSpeed 格式)"""
    print(f"📂 正在加载 checkpoint: {checkpoint_path}")
    
    checkpoint_path = Path(checkpoint_path)
    
    # === 方式 1: Accelerate/DeepSpeed 分布式 checkpoint ===
    if checkpoint_path.is_dir():
        print(f"🔍 检测到目录，尝试加载 Accelerate checkpoint...")
        
        # 查找所有 model 文件
        model_files = list(checkpoint_path.glob("*.bin")) + \
                     list(checkpoint_path.glob("*.pt")) + \
                     list(checkpoint_path.glob("pytorch_model*.bin"))
        
        if not model_files:
            print(f"❌ 在目录中未找到 checkpoint 文件")
            return None
        
        print(f"📦 找到 {len(model_files)} 个 checkpoint 分片")
        
        # 合并所有分片
        merged_state_dict = {}
        for shard_file in sorted(model_files):
            print(f"   - 加载: {shard_file.name}")
            try:
                shard = torch.load(shard_file, map_location='cpu')
                
                # 处理不同的存储格式
                if isinstance(shard, dict):
                    # 可能是 {'model': state_dict} 或直接是 state_dict
                    if 'model' in shard:
                        shard = shard['model']
                    elif 'state_dict' in shard:
                        shard = shard['state_dict']
                    elif 'module' in shard:
                        shard = shard['module']
                
                # 合并到总字典
                for key, value in shard.items():
                    # 移除可能的 'module.' 或 '_orig_mod.' 前缀
                    clean_key = key.replace('module.', '').replace('_orig_mod.', '')
                    merged_state_dict[clean_key] = value
                    
            except Exception as e:
                print(f"⚠️ 跳过文件 {shard_file.name}: {e}")
                continue
        
        if merged_state_dict:
            print(f"✅ 成功加载 Accelerate checkpoint ({len(merged_state_dict)} 个参数)")
            return merged_state_dict
        else:
            print(f"❌ 无法从目录加载任何参数")
            return None
    
    # === 方式 2: 单文件 checkpoint ===
    else:
        # 尝试不同的加载方式
        try:
            # 方式2.1: 直接加载
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # 处理可能的嵌套结构
            if isinstance(state_dict, dict):
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'module' in state_dict:
                    state_dict = state_dict['module']
            
            # 清理 key 前缀
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                clean_key = key.replace('module.', '').replace('_orig_mod.', '')
                cleaned_state_dict[clean_key] = value
            
            print(f"✅ 成功加载 checkpoint (直接模式, {len(cleaned_state_dict)} 个参数)")
            return cleaned_state_dict
            
        except Exception as e:
            print(f"⚠️ 直接加载失败: {e}")
            
            # 方式2.2: 尝试加载压缩的 checkpoint
            try:
                import zstandard as zstd
                with open(checkpoint_path, 'rb') as f:
                    dctx = zstd.ZstdDecompressor()
                    decompressed = dctx.decompress(f.read())
                    import io
                    state_dict = torch.load(io.BytesIO(decompressed), map_location='cpu')
                
                # 处理可能的嵌套结构
                if isinstance(state_dict, dict):
                    if 'model' in state_dict:
                        state_dict = state_dict['model']
                    elif 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                
                # 清理 key 前缀
                cleaned_state_dict = {}
                for key, value in state_dict.items():
                    clean_key = key.replace('module.', '').replace('_orig_mod.', '')
                    cleaned_state_dict[clean_key] = value
                
                print(f"✅ 成功加载 checkpoint (压缩模式, {len(cleaned_state_dict)} 个参数)")
                return cleaned_state_dict
                
            except Exception as e2:
                print(f"❌ 加载失败: {e2}")
                return None


def extract_fusion_weights(checkpoint_path, timesteps, num_dit_layers):
    """
    从 checkpoint 中提取融合权重
    
    Args:
        checkpoint_path: checkpoint 文件路径
        timesteps: 要分析的时间步列表
        num_dit_layers: DiT 层数
    
    Returns:
        weights_dict: {
            'global': np.array (num_timesteps, num_text_layers) 或 None,
            'layer_wise': np.array (num_timesteps, num_dit_layers, num_text_layers) 或 None,
            'config': dict
        }
    """
    state_dict = load_checkpoint(checkpoint_path)
    if state_dict is None:
        return None
    
    # 检测是哪种融合模式
    has_global_module = any('text_fusion_module' in k for k in state_dict.keys())
    has_layer_wise_modules = any('text_fusion_modules' in k for k in state_dict.keys())
    has_global_weight = any('text_fusion_weight' in k for k in state_dict.keys())
    has_layer_wise_weights = any('text_fusion_weights' in k for k in state_dict.keys())
    
    # 判断模式
    use_timestep_adaptive = has_global_module or has_layer_wise_modules
    use_layer_wise = has_layer_wise_modules or has_layer_wise_weights
    
    print(f"\n📊 检测到的融合模式:")
    print(f"   - use_timestep_adaptive_fusion: {use_timestep_adaptive}")
    print(f"   - use_layer_wise_fusion: {use_layer_wise}")
    
    # 检测文本层数
    if has_global_module:
        # 从 weight_generator 的输出层推断
        for key in state_dict.keys():
            if 'text_fusion_module.weight_generator.2.weight' in key:
                num_text_layers = state_dict[key].shape[0]
                break
    elif has_layer_wise_modules:
        for key in state_dict.keys():
            if 'text_fusion_modules.0.weight_generator.2.weight' in key:
                num_text_layers = state_dict[key].shape[0]
                break
    elif has_global_weight:
        num_text_layers = state_dict['text_fusion_weight'].shape[0]
    elif has_layer_wise_weights:
        num_text_layers = state_dict['text_fusion_weights.0'].shape[0]
    else:
        print("❌ 无法检测文本层数")
        return None
    
    print(f"   - text_hidden_states_num: {num_text_layers}")
    print(f"   - dit_num_hidden_layers: {num_dit_layers}")
    
    results = {
        'config': {
            'use_timestep_adaptive': use_timestep_adaptive,
            'use_layer_wise': use_layer_wise,
            'num_text_layers': num_text_layers,
            'num_dit_layers': num_dit_layers,
        }
    }
    
    # === 提取权重 ===
    if use_timestep_adaptive:
        # 模式 2 或 4: 时间自适应融合
        print(f"\n🔄 计算时间自适应权重 (共 {len(timesteps)} 个时间步)...")
        
        if use_layer_wise:
            # 模式 4: 每层独立的时间自适应
            weights = np.zeros((len(timesteps), num_dit_layers, num_text_layers))
            
            for dit_layer_idx in tqdm(range(num_dit_layers), desc="DiT层"):
                # 重建 TimestepWiseFeatureWeighting 模块
                module_state = {}
                prefix = f'text_fusion_modules.{dit_layer_idx}.'
                
                for key in state_dict.keys():
                    if key.startswith(prefix):
                        new_key = key[len(prefix):]
                        module_state[new_key] = state_dict[key]
                
                # 推断 time_embed_dim
                time_embed_dim = module_state['weight_generator.0.weight'].shape[1]
                
                # 创建临时模块
                temp_module = TimestepWiseFeatureWeighting(
                    num_layers=num_text_layers,
                    time_embed_dim=time_embed_dim,
                    feature_dim=2048  # 假设值，不影响权重计算
                )
                temp_module.load_state_dict(module_state)
                temp_module.eval()
                
                # 计算每个时间步的权重
                for t_idx, t in enumerate(timesteps):
                    normalized_t = torch.tensor([t / 1000.0], dtype=torch.float32)
                    t_embed = temp_module._time_embedding(normalized_t)
                    weight = temp_module.weight_generator(t_embed)
                    weights[t_idx, dit_layer_idx, :] = weight.detach().numpy()[0]
            
            results['layer_wise'] = weights
            results['global'] = None
            
        else:
            # 模式 2: 全局时间自适应
            weights = np.zeros((len(timesteps), num_text_layers))
            
            # 重建 TimestepWiseFeatureWeighting 模块
            module_state = {}
            prefix = 'text_fusion_module.'
            
            for key in state_dict.keys():
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    module_state[new_key] = state_dict[key]
            
            # 推断 time_embed_dim
            time_embed_dim = module_state['weight_generator.0.weight'].shape[1]
            
            # 创建临时模块
            temp_module = TimestepWiseFeatureWeighting(
                num_layers=num_text_layers,
                time_embed_dim=time_embed_dim,
                feature_dim=2048  # 假设值
            )
            temp_module.load_state_dict(module_state)
            temp_module.eval()
            
            # 计算每个时间步的权重
            for t_idx, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="时间步"):
                normalized_t = torch.tensor([t / 1000.0], dtype=torch.float32)
                t_embed = temp_module._time_embedding(normalized_t)
                weight = temp_module.weight_generator(t_embed)
                weights[t_idx, :] = weight.detach().numpy()[0]
            
            results['global'] = weights
            results['layer_wise'] = None
    
    else:
        # 模式 1 或 3: 固定权重（不依赖时间步）
        print(f"\n📌 提取固定权重...")
        
        if use_layer_wise:
            # 模式 3: 每层独立的固定权重
            weights = np.zeros((num_dit_layers, num_text_layers))
            
            for dit_layer_idx in range(num_dit_layers):
                raw_weight = state_dict[f'text_fusion_weights.{dit_layer_idx}']
                weight = F.softmax(raw_weight, dim=0).detach().numpy()
                weights[dit_layer_idx, :] = weight
            
            # 对于固定权重，所有时间步都相同
            results['layer_wise'] = np.repeat(weights[np.newaxis, :, :], len(timesteps), axis=0)
            results['global'] = None
            
        else:
            # 模式 1: 全局固定权重
            raw_weight = state_dict['text_fusion_weight']
            weight = F.softmax(raw_weight, dim=0).detach().numpy()
            
            # 对于固定权重，所有时间步都相同
            results['global'] = np.repeat(weight[np.newaxis, :], len(timesteps), axis=0)
            results['layer_wise'] = None
    
    return results


def plot_fusion_weights(weights_dict, timesteps, output_path, dpi=300):
    """
    绘制融合权重可视化图
    
    Args:
        weights_dict: extract_fusion_weights 返回的字典
        timesteps: 时间步列表
        output_path: 输出图片路径
        dpi: 图片分辨率
    """
    config = weights_dict['config']
    num_text_layers = config['num_text_layers']
    use_layer_wise = config['use_layer_wise']
    
    if use_layer_wise:
        # 每层独立融合：为每个 DiT 层画一个子图
        weights = weights_dict['layer_wise']  # (num_timesteps, num_dit_layers, num_text_layers)
        num_dit_layers = weights.shape[1]
        
        # 计算子图布局
        num_cols = min(3, num_dit_layers)
        num_rows = math.ceil(num_dit_layers / num_cols)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 4*num_rows))
        if num_dit_layers == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # 使用不同的颜色和线型
        colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps)))
        linestyles = ['-', '--', '-.', ':'] * ((len(timesteps) // 4) + 1)
        
        for dit_layer_idx in range(num_dit_layers):
            ax = axes[dit_layer_idx]
            
            for t_idx, t in enumerate(timesteps):
                layer_weights = weights[t_idx, dit_layer_idx, :]
                ax.plot(
                    range(num_text_layers),
                    layer_weights,
                    marker='o',
                    linestyle=linestyles[t_idx],
                    color=colors[t_idx],
                    linewidth=2,
                    markersize=6,
                    label=f't={t}',
                    alpha=0.8
                )
            
            ax.set_xlabel('Text Layer Index', fontsize=12, fontweight='bold')
            ax.set_ylabel('Weight (Softmax)', fontsize=12, fontweight='bold')
            ax.set_title(f'DiT Layer {dit_layer_idx}', fontsize=14, fontweight='bold')
            ax.set_xticks(range(num_text_layers))
            ax.set_xticklabels([f'L{i}' for i in range(num_text_layers)])
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=10, ncol=2)
            ax.set_ylim([0, 1])
        
        # 隐藏多余的子图
        for idx in range(num_dit_layers, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(
            f'Layer-wise Fusion Weights across Timesteps\n'
            f'({"Adaptive" if config["use_timestep_adaptive"] else "Fixed"})',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )
        
    else:
        # 全局共享融合：画一个图
        weights = weights_dict['global']  # (num_timesteps, num_text_layers)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 使用不同的颜色和线型
        colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps)))
        linestyles = ['-', '--', '-.', ':'] * ((len(timesteps) // 4) + 1)
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p'] * ((len(timesteps) // 8) + 1)
        
        for t_idx, t in enumerate(timesteps):
            layer_weights = weights[t_idx, :]
            ax.plot(
                range(num_text_layers),
                layer_weights,
                marker=markers[t_idx],
                linestyle=linestyles[t_idx],
                color=colors[t_idx],
                linewidth=2.5,
                markersize=8,
                label=f'Timestep {t}',
                alpha=0.85
            )
        
        ax.set_xlabel('Text Layer Index', fontsize=14, fontweight='bold')
        ax.set_ylabel('Fusion Weight (After Softmax)', fontsize=14, fontweight='bold')
        ax.set_title(
            f'Global Fusion Weights across Timesteps\n'
            f'({"Timestep-Adaptive" if config["use_timestep_adaptive"] else "Fixed"})',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        ax.set_xticks(range(num_text_layers))
        ax.set_xticklabels([f'Layer {i}' for i in range(num_text_layers)])
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax.legend(loc='best', fontsize=12, ncol=2, framealpha=0.9)
        ax.set_ylim([0, 1.0])
        
        # 添加水平参考线
        ax.axhline(y=1.0/num_text_layers, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='Uniform')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\n✅ 可视化图已保存到: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='可视化 AdaFuseDiT 模型的融合权重'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='/ytech_m2v5_hdd/workspace/kling_mm/libozhou/feature_combination/output/256-AdaFuseDiT-timewise/25000/mp_rank_00_model_states.pt',
        help='Checkpoint 文件路径 (例如: checkpoints/model/ema.pt.zst 或 model.pt)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='visual/fusion_weights_visualization.png',
        help='输出图片路径'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        nargs='+',
        default=[0, 100, 200, 300, 500, 700, 900, 999],
        help='要分析的时间步列表 (空格分隔，默认: 0 100 200 300 500 700 900 999)'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=18,
        help='DiT 层数 (默认: 18)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='图片分辨率 (默认: 300)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎨 AdaFuseDiT 融合权重可视化工具")
    print("=" * 60)
    print(f"📁 Checkpoint: {args.checkpoint_path}")
    print(f"🖼️  输出路径: {args.output_path}")
    print(f"⏱️  时间步: {args.timesteps}")
    print(f"🔢 DiT 层数: {args.num_layers}")
    print("=" * 60)
    
    # 提取权重
    weights_dict = extract_fusion_weights(
        args.checkpoint_path,
        args.timesteps,
        args.num_layers
    )
    
    if weights_dict is None:
        print("\n❌ 权重提取失败，请检查 checkpoint 路径")
        return
    
    # 绘制可视化
    plot_fusion_weights(
        weights_dict,
        args.timesteps,
        args.output_path,
        dpi=args.dpi
    )
    
    print("\n" + "=" * 60)
    print("✨ 完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
