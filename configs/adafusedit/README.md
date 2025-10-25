# AdaFuseDiT 配置说明

## 四种融合模式

AdaFuseDiT 支持四种文本特征融合模式，通过两个布尔参数组合控制：

| 模式 | `use_timestep_adaptive_fusion` | `use_layer_wise_fusion` | 描述 | 参数量 |
|------|-------------------------------|------------------------|------|--------|
| **模式 1** | `false` | `false` | 全局固定可学习权重 | 最少 |
| **模式 2** | `true` | `false` | 全局时间自适应权重 | 中等 |
| **模式 3** | `false` | `true` | 每层独立固定权重 | 中等 |
| **模式 4** | `true` | `true` | 每层独立时间自适应权重 | 最多（推荐）|

## 配置示例

### 模式 1: 全局固定权重（最简单）
```yaml
model:
  name: "AdaFuseDiT"
  text_hidden_states_num: 4
  use_timestep_adaptive_fusion: false
  use_layer_wise_fusion: false
  # adaptive_fusion_time_embed_dim: 128  # 此模式不需要
```

### 模式 2: 全局时间自适应权重
```yaml
model:
  name: "AdaFuseDiT"
  text_hidden_states_num: 4
  use_timestep_adaptive_fusion: true
  use_layer_wise_fusion: false
  adaptive_fusion_time_embed_dim: 128
```

### 模式 3: 每层独立固定权重
```yaml
model:
  name: "AdaFuseDiT"
  text_hidden_states_num: 4
  use_timestep_adaptive_fusion: false
  use_layer_wise_fusion: true
  # adaptive_fusion_time_embed_dim: 128  # 此模式不需要
```

### 模式 4: 每层独立时间自适应权重（推荐，最灵活）
```yaml
model:
  name: "AdaFuseDiT"
  text_hidden_states_num: 4
  use_timestep_adaptive_fusion: true
  use_layer_wise_fusion: true
  adaptive_fusion_time_embed_dim: 128
```

## 关键参数说明

- **text_hidden_states_num**: 使用文本编码器的层数（例如：4 表示使用最后 4 层）
- **use_timestep_adaptive_fusion**: 是否使用基于 MLP 的时间自适应融合
- **use_layer_wise_fusion**: 是否为每个 DiT 层独立学习融合权重
- **adaptive_fusion_time_embed_dim**: 时间嵌入向量维度（仅时间自适应模式需要）

## Pipeline 使用

### 训练时
```python
from diffusion.pipelines import AdaFuseDiTPipeline

# Pipeline 会自动根据 config.text_hidden_states_num 提取多层特征
# 无需手动处理
```

### 推理时
```python
from diffusion.pipelines import AdaFuseDiTPipeline
from diffusion.models import AdaFuseDiT

# 加载模型
model = AdaFuseDiT.from_pretrained("your-checkpoint")

# 创建 pipeline
pipe = AdaFuseDiTPipeline(
    transformer=model,
    scheduler=scheduler,
    vae=vae,
    tokenizer=tokenizer,
    llm=llm
)

# 生成图像
images = pipe(
    prompt="A beautiful sunset over the ocean",
    height=512,
    width=512,
    num_inference_steps=28,
    guidance_scale=7.0
)
```

## 与标准 DiT 的区别

1. **多层文本特征**：AdaFuseDiT 可以使用多个文本编码器层
2. **自适应融合**：可以根据扩散时间步动态调整融合权重
3. **层级融合**：每个 DiT 层可以有独立的融合策略

## 建议

- 初始实验：使用模式 4（最灵活，效果最好）
- 资源受限：使用模式 1（参数最少）
- 中等方案：使用模式 2 或 3

## 兼容性

- AdaFuseDiT 向后兼容：当 `text_hidden_states_num=1` 时，行为与标准 DiT 相同
- 可以直接替换现有的 DiT 配置文件中的 `name: "DiT"` 为 `name: "AdaFuseDiT"`
