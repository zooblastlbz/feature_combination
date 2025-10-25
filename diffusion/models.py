import functools
import math
from typing import Optional

from diffusers.models.embeddings import PatchEmbed, Timesteps
from diffusers.utils import is_torch_xla_available
from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    GemmaConfig,
    GemmaForCausalLM,
    GemmaModel,
    Gemma2Config,
    Gemma2ForCausalLM,
    Gemma2Model,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
)
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gemma.modeling_gemma import (
    apply_rotary_pos_emb,
    GemmaRMSNorm,
    GemmaRotaryEmbedding,
    repeat_kv,
)

from .configs import DiTConfig, FuseDiTConfig
from .modules import AdaLayerNormOut, DiTLayer

if is_torch_xla_available():
    ACCEL = "xla"
elif torch.cuda.is_available():
    ACCEL = "cuda"
else:
    ACCEL = "cpu"


class TimestepWiseFeatureWeighting(nn.Module):
    """
    基于MLP的时序感知特征加权模块。

    该模块接收一个时间步（timestep）t 和一组来自文本编码器不同层的特征，
    然后使用一个小型MLP网络学习一个最优的加权策略，将这些特征组合成单一的特征表示。
    """
    def __init__(self, num_layers: int, time_embed_dim: int, feature_dim: int):
        """
        初始化模块。

        Args:
            num_layers (int): 需要进行加权组合的特征层数。
            time_embed_dim (int): 用于表示时间步t的嵌入向量维度。
            feature_dim (int): 输入特征的维度，用于 layer norm。
        """
        super().__init__()
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim
        self.feature_dim = feature_dim

        # 定义用于生成权重的MLP网络
        # hidden_dim 自动设置为 time_embed_dim 的 4 倍
        hidden_dim = time_embed_dim * 4
        self.weight_generator = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_layers),
            nn.Softmax(dim=-1)
        )

    def _time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        使用正弦/余弦编码将时间步t转换为高维向量。

        Args:
            timesteps (torch.Tensor): 批量的timesteps，shape: (batch_size,)。

        Returns:
            torch.Tensor: 编码后的时间嵌入向量，shape: (batch_size, time_embed_dim)。
        """
        timesteps = timesteps.unsqueeze(-1)
        
        half_dim = self.time_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(timesteps.device)
        
        args = timesteps * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.time_embed_dim % 2 != 0:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding

    def forward(self, hidden_states: list, timesteps: torch.Tensor):
        """
        模块的前向传播逻辑。

        Args:
            hidden_states (list[torch.Tensor]): 包含多层特征的列表。
                列表中每个元素的 shape 都是 (batch_size, seq_len, feature_dim)。
            timesteps (torch.Tensor): 扩散模型的timesteps。
                shape 为 (batch_size,)。

        Returns:
            tuple: (combined_features, weights)
                - combined_features: 加权组合后的特征，shape: (batch_size, seq_len, feature_dim)。
                - weights: 生成的权重，shape: (batch_size, num_layers)。
        """
        if len(hidden_states) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} hidden states, but got {len(hidden_states)}")

        # 在聚合前对每层应用 layer norm（不可训练）
        # 使用 F.layer_norm 以避免引入可训练参数
        normalized_hidden_states = [
            F.layer_norm(h, normalized_shape=(self.feature_dim,))
            for h in hidden_states
        ]

        # 堆叠隐藏状态: (num_layers, batch_size, seq_len, feature_dim)
        stacked_hidden_states = torch.stack(normalized_hidden_states, dim=0)

        # 生成时序权重
        t_embed = self._time_embedding(timesteps)
        weights = self.weight_generator(t_embed)

        # 加权求和
        broadcastable_weights = weights.T.unsqueeze(-1).unsqueeze(-1)
        weighted_states = stacked_hidden_states * broadcastable_weights
        combined_features = torch.sum(weighted_states, dim=0)

        return combined_features, weights


def get_llm(model: str, base_config: PretrainedConfig):
    if isinstance(base_config, GemmaConfig):
        return GemmaForCausalLM.from_pretrained(model).model
    elif isinstance(base_config, Gemma2Config):
        return Gemma2ForCausalLM.from_pretrained(model).model
    elif isinstance(base_config, PaliGemmaConfig):
        return PaliGemmaForConditionalGeneration.from_pretrained(model).language_model.model
    else:
        raise ValueError(f"Unknown model: {model}")


def update_self_attention_mask(llm_attention_mask: torch.LongTensor, dit_sequence_length: int, use_cache: bool, device, dtype):
    llm_sequence_length = llm_attention_mask.shape[1]
    sequence_length = dit_sequence_length + llm_sequence_length
    min_dtype = torch.finfo(dtype).min

    attention_mask = torch.full(
        (llm_attention_mask.shape[0], 1, sequence_length, sequence_length),
        fill_value=min_dtype,
        device=device,
        dtype=dtype,
    )

    attention_mask[:, :, :llm_sequence_length, :llm_sequence_length] = torch.triu(attention_mask[:, :, :llm_sequence_length, :llm_sequence_length], diagonal=1) # Causal mask for LLM
    attention_mask[:, :, llm_sequence_length:, :] = 0 # Bi-directional mask for DiT

    padding_mask = attention_mask[:, :, :, :llm_sequence_length] + repeat(llm_attention_mask, "b n -> b 1 1 n")
    padding_mask = padding_mask == 0
    attention_mask[:, :, :, :llm_sequence_length] = attention_mask[:, :, :, :llm_sequence_length].masked_fill(padding_mask, min_dtype)  # Padding mask for LLM
   
    if use_cache:
        attention_mask = attention_mask[:, :, llm_sequence_length:, :]

    return attention_mask


def update_cross_attention_mask(llm_attention_mask: torch.LongTensor, dit_sequence_length: int, joint: bool, device, dtype):
    min_dtype = torch.finfo(dtype).min

    if joint:
        llm_attention_mask = torch.cat([llm_attention_mask, torch.ones(llm_attention_mask.shape[0], dit_sequence_length, device=device, dtype=dtype)], dim=-1)

    cross_attention_mask = repeat(llm_attention_mask, "b n -> b 1 s n", s=dit_sequence_length).to(dtype=torch.bool)
    cross_attention_mask = torch.zeros_like(cross_attention_mask, device=device, dtype=dtype).masked_fill(cross_attention_mask.logical_not(), min_dtype)

    return cross_attention_mask


def unpatchify(dit_hidden_states: torch.FloatTensor, height: int, width: int, patch_size: int):
    # unpatchify
    output = rearrange(
        dit_hidden_states,
        "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
        h=height,
        w=width,
        p1=patch_size,
        p2=patch_size,
    )

    return output


class DiT(PreTrainedModel):
    """
    Diffusion Transformer Model.
    """

    config_class = DiTConfig
    supports_gradient_checkpointing = True
    _supports_sdpa = True  # * Make Transformers happy

    def __init__(self, config: DiTConfig):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [DiTLayer(config) for _ in range(config.dit_num_hidden_layers)]
        )

        self.patch_embed = PatchEmbed(
            height=config.sample_size,
            width=config.sample_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.dit_hidden_size,
            pos_embed_type=None if not config.pos_embed == "ape" else "sincos",
            pos_embed_max_size=config.pos_embed_max_size,
        )

        self.rotary_emb = GemmaRotaryEmbedding(
            config.base_config.head_dim,
            config.base_config.max_position_embeddings,
            config.base_config.rope_theta,
        )
        if config.pos_embed == "2d-rope":
            self.rotary_emb_half = GemmaRotaryEmbedding(
                config.base_config.head_dim // 2,
                config.base_config.max_position_embeddings,
                config.base_config.rope_theta,
            )
        
        if config.timestep_conditioning is not None:
            self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.timestep_embedder = nn.Sequential(
                nn.Linear(256, config.dit_hidden_size),
                nn.SiLU(),
                nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
            )
        if config.timestep_conditioning == "adaln-single":
            self.t_block = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.dit_hidden_size, 6 * config.dit_hidden_size),
            )
            self.scale_shift_table = nn.Parameter(torch.randn(2 * config.dit_hidden_size) / config.dit_hidden_size ** 0.5)

        if config.text_modulation_embeds_dim is not None:
            self.condition_embedder = nn.Sequential(
                GemmaRMSNorm(config.text_modulation_embeds_dim, eps=config.base_config.rms_norm_eps),
                nn.Linear(config.text_modulation_embeds_dim, config.dit_hidden_size),
                nn.SiLU(),
                nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
            )
        if config.model_type == "DiT":
            self.context_embedder = nn.Sequential(
                GemmaRMSNorm(config.text_hidden_size, eps=config.base_config.rms_norm_eps),
                nn.Linear(config.text_hidden_size, config.dit_hidden_size),
            )

        if config.timestep_conditioning == "adaln-zero":
            self.norm_out = AdaLayerNormOut(config)
        else:
            self.norm_out = GemmaRMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)
        self.proj_out = nn.Linear(
            config.dit_hidden_size,
            config.patch_size * config.patch_size * config.out_channels,
            bias=True,
        )

        self.gradient_checkpointing = False

        self.initialize_weights()

    def prepare_hidden_states(self, hidden_states: torch.FloatTensor, temb: torch.FloatTensor, height: int, width: int):
        if not self.config.pos_embed == "ape":
            if self.config.timestep_conditioning == "addition":
                hidden_states += temb.unsqueeze(1)

            if self.config.pos_embed == "m-rope" or self.config.pos_embed == "2d-rope":
                h_position_ids = torch.arange(height, device=hidden_states.device).repeat_interleave(width)
                h_position_ids = repeat(h_position_ids, "s -> b s", b=hidden_states.shape[0])
                w_position_ids = torch.arange(width, device=hidden_states.device).repeat(height)
                w_position_ids = repeat(w_position_ids, "s -> b s", b=hidden_states.shape[0])

                if self.config.pos_embed == "m-rope":
                    h_cos, h_sin = self.rotary_emb(hidden_states, h_position_ids)
                    w_cos, w_sin = self.rotary_emb(hidden_states, w_position_ids)
                    cos = torch.stack([h_cos, w_cos], dim=0)
                    cos = torch.cat([m[i % 2] for i, m in enumerate(cos.chunk(4, dim=-1))], dim=-1)
                    sin = torch.stack([h_sin, w_sin], dim=0)
                    sin = torch.cat([m[i % 2] for i, m in enumerate(sin.chunk(4, dim=-1))], dim=-1)
                elif self.config.pos_embed == "2d-rope":
                    h_cos, h_sin = self.rotary_emb_half(hidden_states, h_position_ids)
                    w_cos, w_sin = self.rotary_emb_half(hidden_states, w_position_ids)
                    cos = torch.stack([h_cos, w_cos], dim=2)
                    cos = torch.cat([rearrange(m, "b s n d -> b s (n d)") for m in cos.chunk(2, dim=-1)], dim=-1)
                    sin = torch.stack([h_sin, w_sin], dim=2)
                    sin = torch.cat([rearrange(m, "b s n d -> b s (n d)") for m in sin.chunk(2, dim=-1)], dim=-1)

                pos_embed = (cos, sin)
            else:
                position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)

                pos_embed = self.rotary_emb(hidden_states, position_ids)
        else:
            if self.config.timestep_conditioning == "addition":
                hidden_states += temb.unsqueeze(1)

            pos_embed = None

        return hidden_states, pos_embed

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor,
        text_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        height, width = hidden_states.shape[-2:]
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = self.patch_embed(hidden_states)

        if self.config.timestep_conditioning is not None:
            timestep = self.time_proj(timestep).to(hidden_states.dtype)
            timestep = self.timestep_embedder(timestep)

            if self.config.timestep_conditioning == "adaln-single":
                temb = self.t_block(timestep)
            else:
                temb = timestep
        else:
            temb = torch.zeros_like(hidden_states[:, 0, :])

        hidden_states, pos_embed = self.prepare_hidden_states(hidden_states, temb, height, width)
        text_hidden_states = self.context_embedder(text_hidden_states)

        cross_attention_mask = update_cross_attention_mask(
            attention_mask,
            hidden_states.shape[1],
            self.config.attention == "self",
            hidden_states.device,
            torch.float32 if ACCEL == "xla" else text_hidden_states.dtype,
        )

        for layer_index in range(self.config.dit_num_hidden_layers):
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    self.layers[layer_index].__call__,
                    hidden_states,
                    temb,
                    pos_embed,
                    text_hidden_states,
                    cross_attention_mask,
                )

            else:
                hidden_states = self.layers[layer_index](hidden_states, temb, pos_embed, text_hidden_states, cross_attention_mask)

        if self.config.timestep_conditioning == "adaln-zero":
            hidden_states = self.norm_out(hidden_states, temb)
        else:
            hidden_states = self.norm_out(hidden_states)
        if self.config.timestep_conditioning == "adaln-single":
            shift, scale = (self.scale_shift_table + repeat(timestep, "b d -> b (2 d)")).chunk(2, dim=1)
            hidden_states = hidden_states * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out(hidden_states)

        output = unpatchify(hidden_states, height, width, patch_size)

        return output

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks and output layers:
        if self.config.timestep_conditioning == "adaln-zero":
            for layer in self.layers:
                nn.init.constant_(layer.input_layernorm.linear.weight, 0)
                nn.init.constant_(layer.input_layernorm.linear.bias, 0)

        if self.config.timestep_conditioning == "adaln-zero":
            nn.init.constant_(self.norm_out.linear.weight, 0)
            nn.init.constant_(self.norm_out.linear.bias, 0)
        elif self.config.timestep_conditioning == "adaln-single":
            nn.init.normal_(self.t_block[1].weight, std=self.config.base_config.initializer_range)
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        # Initialize timestep embedding MLP:
        if self.config.timestep_conditioning is not None:
            nn.init.normal_(self.timestep_embedder[0].weight, std=self.config.base_config.initializer_range)
            nn.init.normal_(self.timestep_embedder[2].weight, std=self.config.base_config.initializer_range)

        if self.config.text_modulation_embeds_dim is not None:
            # Initialize condition embedding MLP:
            nn.init.normal_(self.condition_embedder[1].weight, std=self.config.base_config.initializer_range)
            nn.init.normal_(self.condition_embedder[3].weight, std=self.config.base_config.initializer_range)
        if self.config.model_type == "DiT":
            # Initialize context embedding MLP:
            nn.init.normal_(self.context_embedder[1].weight, std=self.config.base_config.initializer_range)

        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = { "use_reentrant": True }

        gradient_checkpointing_func = functools.partial(torch.utils.checkpoint.checkpoint, **gradient_checkpointing_kwargs)
        
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())


class AdaFuseDiT(PreTrainedModel):
    """
    Adaptive Fuse Diffusion Transformer Model.
    
    AdaFuseDiT 在标准 DiT 的基础上，集成了时序感知的特征加权模块。
    该模型能够根据扩散过程的不同时间步，动态地融合来自文本编码器多个层的特征，
    实现更加灵活和自适应的文本-图像条件化策略。
    
    支持四种融合模式（通过配置参数组合）：
    1. 全局固定权重：use_timestep_adaptive_fusion=False, use_layer_wise_fusion=False
    2. 全局时间自适应权重：use_timestep_adaptive_fusion=True, use_layer_wise_fusion=False
    3. 每层独立固定权重：use_timestep_adaptive_fusion=False, use_layer_wise_fusion=True
    4. 每层独立时间自适应权重：use_timestep_adaptive_fusion=True, use_layer_wise_fusion=True
    """

    config_class = DiTConfig
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def __init__(self, config: DiTConfig):
        super().__init__(config)

        # 标准 DiT 层
        self.layers = nn.ModuleList(
            [DiTLayer(config) for _ in range(config.dit_num_hidden_layers)]
        )

        self.patch_embed = PatchEmbed(
            height=config.sample_size,
            width=config.sample_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.dit_hidden_size,
            pos_embed_type=None if not config.pos_embed == "ape" else "sincos",
            pos_embed_max_size=config.pos_embed_max_size,
        )

        self.rotary_emb = GemmaRotaryEmbedding(
            config.base_config.head_dim,
            config.base_config.max_position_embeddings,
            config.base_config.rope_theta,
        )
        if config.pos_embed == "2d-rope":
            self.rotary_emb_half = GemmaRotaryEmbedding(
                config.base_config.head_dim // 2,
                config.base_config.max_position_embeddings,
                config.base_config.rope_theta,
            )
        
        if config.timestep_conditioning is not None:
            self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.timestep_embedder = nn.Sequential(
                nn.Linear(256, config.dit_hidden_size),
                nn.SiLU(),
                nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
            )
        if config.timestep_conditioning == "adaln-single":
            self.t_block = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.dit_hidden_size, 6 * config.dit_hidden_size),
            )
            self.scale_shift_table = nn.Parameter(torch.randn(2 * config.dit_hidden_size) / config.dit_hidden_size ** 0.5)

        if config.text_modulation_embeds_dim is not None:
            self.condition_embedder = nn.Sequential(
                GemmaRMSNorm(config.text_modulation_embeds_dim, eps=config.base_config.rms_norm_eps),
                nn.Linear(config.text_modulation_embeds_dim, config.dit_hidden_size),
                nn.SiLU(),
                nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
            )
        
        # === AdaFuseDiT 核心：根据配置创建文本特征融合模块 ===
        if config.text_hidden_states_num > 1:
            if config.use_layer_wise_fusion:
                # 每层独立融合
                if config.use_timestep_adaptive_fusion:
                    # 模式 4: 每层独立的 TimestepWiseFeatureWeighting
                    self.text_fusion_modules = nn.ModuleList([
                        TimestepWiseFeatureWeighting(
                            num_layers=config.text_hidden_states_num,
                            time_embed_dim=config.adaptive_fusion_time_embed_dim,
                            feature_dim=config.text_hidden_size
                        )
                        for _ in range(config.dit_num_hidden_layers)
                    ])
                else:
                    # 模式 3: 每层独立的可学习权重
                    self.text_fusion_weights = nn.ParameterList([
                        nn.Parameter(torch.randn(config.text_hidden_states_num))
                        for _ in range(config.dit_num_hidden_layers)
                    ])
            else:
                # 全局共享融合
                if config.use_timestep_adaptive_fusion:
                    # 模式 2: 全局 TimestepWiseFeatureWeighting
                    self.text_fusion_module = TimestepWiseFeatureWeighting(
                        num_layers=config.text_hidden_states_num,
                        time_embed_dim=config.adaptive_fusion_time_embed_dim,
                        feature_dim=config.text_hidden_size
                    )
                else:
                    # 模式 1: 全局可学习权重
                    self.text_fusion_weight = nn.Parameter(
                        torch.randn(config.text_hidden_states_num)
                    )
        
        # 用于将组合后的文本特征映射到 DiT 的隐藏维度
        self.context_embedder = nn.Sequential(
            GemmaRMSNorm(config.text_hidden_size, eps=config.base_config.rms_norm_eps),
            nn.Linear(config.text_hidden_size, config.dit_hidden_size),
        )

        if config.timestep_conditioning == "adaln-zero":
            self.norm_out = AdaLayerNormOut(config)
        else:
            self.norm_out = GemmaRMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)
        self.proj_out = nn.Linear(
            config.dit_hidden_size,
            config.patch_size * config.patch_size * config.out_channels,
            bias=True,
        )

        self.gradient_checkpointing = False

        self.initialize_weights()

    def _fuse_text_features(
        self, 
        text_hidden_states: list, 
        timestep: torch.Tensor, 
        layer_idx: Optional[int] = None
    ):
        """
        统一的文本特征融合接口。
        
        Args:
            text_hidden_states: list of tensors, 每个 shape (batch_size, seq_len, hidden_size)
            timestep: shape (batch_size,)
            layer_idx: 当前 DiT 层的索引（仅在 use_layer_wise_fusion=True 时使用）
        
        Returns:
            fused_text: shape (batch_size, seq_len, hidden_size)
        """
        if self.config.text_hidden_states_num == 1:
            # 不使用多层融合，直接返回单层特征
            return text_hidden_states[0] if isinstance(text_hidden_states, list) else text_hidden_states
        
        if self.config.use_layer_wise_fusion:
            # 每层独立融合
            if self.config.use_timestep_adaptive_fusion:
                # 模式 4: 使用该层的 TimestepWiseFeatureWeighting
                normalized_timestep = timestep.float() / 1000.0
                fused_text, _ = self.text_fusion_modules[layer_idx](
                    text_hidden_states,
                    normalized_timestep
                )
            else:
                # 模式 3: 使用该层的可学习权重
                # 先对每层进行 layer norm（不可训练）
                normalized_hidden_states = [
                    F.layer_norm(h, normalized_shape=(self.config.text_hidden_size,))
                    for h in text_hidden_states
                ]
                weights = F.softmax(self.text_fusion_weights[layer_idx], dim=0)
                stacked = torch.stack(normalized_hidden_states, dim=0)
                fused_text = torch.sum(
                    stacked * weights.view(-1, 1, 1, 1),
                    dim=0
                )
        else:
            # 全局共享融合
            if self.config.use_timestep_adaptive_fusion:
                # 模式 2: 使用全局 TimestepWiseFeatureWeighting
                normalized_timestep = timestep.float() / 1000.0
                fused_text, _ = self.text_fusion_module(
                    text_hidden_states,
                    normalized_timestep
                )
            else:
                # 模式 1: 使用全局可学习权重
                # 先对每层进行 layer norm（不可训练）
                normalized_hidden_states = [
                    F.layer_norm(h, normalized_shape=(self.config.text_hidden_size,))
                    for h in text_hidden_states
                ]
                weights = F.softmax(self.text_fusion_weight, dim=0)
                stacked = torch.stack(normalized_hidden_states, dim=0)
                fused_text = torch.sum(
                    stacked * weights.view(-1, 1, 1, 1),
                    dim=0
                )
        
        return fused_text

    def prepare_hidden_states(self, hidden_states: torch.FloatTensor, temb: torch.FloatTensor, height: int, width: int):
        if not self.config.pos_embed == "ape":
            if self.config.timestep_conditioning == "addition":
                hidden_states += temb.unsqueeze(1)

            if self.config.pos_embed == "m-rope" or self.config.pos_embed == "2d-rope":
                h_position_ids = torch.arange(height, device=hidden_states.device).repeat_interleave(width)
                h_position_ids = repeat(h_position_ids, "s -> b s", b=hidden_states.shape[0])
                w_position_ids = torch.arange(width, device=hidden_states.device).repeat(height)
                w_position_ids = repeat(w_position_ids, "s -> b s", b=hidden_states.shape[0])

                if self.config.pos_embed == "m-rope":
                    h_cos, h_sin = self.rotary_emb(hidden_states, h_position_ids)
                    w_cos, w_sin = self.rotary_emb(hidden_states, w_position_ids)
                    cos = torch.stack([h_cos, w_cos], dim=0)
                    cos = torch.cat([m[i % 2] for i, m in enumerate(cos.chunk(4, dim=-1))], dim=-1)
                    sin = torch.stack([h_sin, w_sin], dim=0)
                    sin = torch.cat([m[i % 2] for i, m in enumerate(sin.chunk(4, dim=-1))], dim=-1)
                elif self.config.pos_embed == "2d-rope":
                    h_cos, h_sin = self.rotary_emb_half(hidden_states, h_position_ids)
                    w_cos, w_sin = self.rotary_emb_half(hidden_states, w_position_ids)
                    cos = torch.stack([h_cos, w_cos], dim=2)
                    cos = torch.cat([rearrange(m, "b s n d -> b s (n d)") for m in cos.chunk(2, dim=-1)], dim=-1)
                    sin = torch.stack([h_sin, w_sin], dim=2)
                    sin = torch.cat([rearrange(m, "b s n d -> b s (n d)") for m in sin.chunk(2, dim=-1)], dim=-1)

                pos_embed = (cos, sin)
            else:
                position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
                pos_embed = self.rotary_emb(hidden_states, position_ids)
        else:
            if self.config.timestep_conditioning == "addition":
                hidden_states += temb.unsqueeze(1)
            pos_embed = None

        return hidden_states, pos_embed

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor,
        text_hidden_states: Optional[list] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        前向传播。

        Args:
            hidden_states: 图像隐藏状态，shape (batch_size, channels, height, width)
            timestep: 扩散时间步，shape (batch_size,)
            text_hidden_states: 文本隐藏状态
                - 如果 text_hidden_states_num > 1: list of tensors, 每个 shape (batch_size, seq_len, hidden_size)
                - 否则: single tensor, shape (batch_size, seq_len, hidden_size)
            attention_mask: 注意力掩码，shape (batch_size, seq_len)
        """
        height, width = hidden_states.shape[-2:]
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = self.patch_embed(hidden_states)

        if self.config.timestep_conditioning is not None:
            timestep_embed = self.time_proj(timestep).to(hidden_states.dtype)
            timestep_embed = self.timestep_embedder(timestep_embed)

            if self.config.timestep_conditioning == "adaln-single":
                temb = self.t_block(timestep_embed)
            else:
                temb = timestep_embed
        else:
            temb = torch.zeros_like(hidden_states[:, 0, :])

        hidden_states, pos_embed = self.prepare_hidden_states(hidden_states, temb, height, width)

        cross_attention_mask = update_cross_attention_mask(
            attention_mask,
            hidden_states.shape[1],
            self.config.attention == "self",
            hidden_states.device,
            torch.float32 if ACCEL == "xla" else hidden_states.dtype,
        )

        # 每层独立融合文本特征
        for layer_idx in range(self.config.dit_num_hidden_layers):
            # 为当前层融合文本特征
            fused_text = self._fuse_text_features(
                text_hidden_states,
                timestep,
                layer_idx=layer_idx if self.config.use_layer_wise_fusion else None
            )
            
            # 映射到 DiT 的隐藏维度
            fused_text = self.context_embedder(fused_text)
            
            # 执行 DiT 层
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    self.layers[layer_idx].__call__,
                    hidden_states,
                    temb,
                    pos_embed,
                    fused_text,
                    cross_attention_mask,
                )
            else:
                hidden_states = self.layers[layer_idx](
                    hidden_states,
                    temb,
                    pos_embed,
                    fused_text,
                    cross_attention_mask
                )

        if self.config.timestep_conditioning == "adaln-zero":
            hidden_states = self.norm_out(hidden_states, temb)
        else:
            hidden_states = self.norm_out(hidden_states)
        if self.config.timestep_conditioning == "adaln-single":
            shift, scale = (self.scale_shift_table + repeat(timestep_embed, "b d -> b (2 d)")).chunk(2, dim=1)
            hidden_states = hidden_states * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out(hidden_states)

        output = unpatchify(hidden_states, height, width, patch_size)

        return output

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        if self.config.timestep_conditioning == "adaln-zero":
            for layer in self.layers:
                nn.init.constant_(layer.input_layernorm.linear.weight, 0)
                nn.init.constant_(layer.input_layernorm.linear.bias, 0)

        if self.config.timestep_conditioning == "adaln-zero":
            nn.init.constant_(self.norm_out.linear.weight, 0)
            nn.init.constant_(self.norm_out.linear.bias, 0)
        elif self.config.timestep_conditioning == "adaln-single":
            nn.init.normal_(self.t_block[1].weight, std=self.config.base_config.initializer_range)
        
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        if self.config.timestep_conditioning is not None:
            nn.init.normal_(self.timestep_embedder[0].weight, std=self.config.base_config.initializer_range)
            nn.init.normal_(self.timestep_embedder[2].weight, std=self.config.base_config.initializer_range)

        if self.config.text_modulation_embeds_dim is not None:
            nn.init.normal_(self.condition_embedder[1].weight, std=self.config.base_config.initializer_range)
            nn.init.normal_(self.condition_embedder[3].weight, std=self.config.base_config.initializer_range)
        
        nn.init.normal_(self.context_embedder[1].weight, std=self.config.base_config.initializer_range)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = { "use_reentrant": True }
        gradient_checkpointing_func = functools.partial(torch.utils.checkpoint.checkpoint, **gradient_checkpointing_kwargs)
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())


class FuseDiT(PreTrainedModel):
    """
    FuseDiT Model.
    """

    config_class = FuseDiTConfig
    supports_gradient_checkpointing = True
    _supports_sdpa = True  # * Make Transformers happy

    def __init__(self, config: FuseDiTConfig):
        super().__init__(config)

        if config.base_config.model_type == "gemma":
            self.llm = GemmaModel(config.base_config)
        elif config.base_config.model_type == "gemma2":
            self.llm = Gemma2Model(config.base_config)
        else:
            raise ValueError(f"Model type {config.base_config.model_type} not supported.")

        self.dit = DiT(config)

        self.num_key_value_groups = config.base_config.num_attention_heads // config.base_config.num_key_value_heads

        self.gradient_checkpointing = False

    def shared_transformer_layer(
        self,
        index: int,
        dit_hidden_states: torch.FloatTensor,
        llm_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        pos_embed: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        cross_attention_mask: Optional[torch.FloatTensor],
        past_key_values: Optional[DynamicCache] = None,
    ):
        """
        Joint Transformer Layer.
        """

        if self.config.timestep_conditioning == "adaln-zero":
            (
                norm_dit_hidden_states,
                dit_gate_msa,
                dit_shift_mlp,
                dit_scale_mlp,
                dit_gate_mlp,
            ) = self.dit.layers[index - self.config.initial_layers].input_layernorm(dit_hidden_states, emb=temb)
            dit_scale_msa = dit_shift_msa = torch.zeros_like(temb) # Dummy values.
        else:
            norm_dit_hidden_states = self.dit.layers[index - self.config.initial_layers].input_layernorm(dit_hidden_states)
            if self.config.timestep_conditioning == "adaln-single":
                (
                    dit_shift_msa,
                    dit_scale_msa,
                    dit_gate_msa,
                    dit_shift_mlp,
                    dit_scale_mlp,
                    dit_gate_mlp
                ) = (self.dit.layers[index - self.config.initial_layers].scale_shift_table + temb).chunk(6, dim=1)
            else:
                dit_shift_msa = dit_scale_msa = dit_shift_mlp = dit_scale_mlp = torch.zeros_like(temb)
                dit_gate_msa = dit_gate_mlp = torch.ones_like(temb)

        norm_dit_hidden_states = norm_dit_hidden_states * (1 + dit_scale_msa[:, None]) + dit_shift_msa[:, None]
        if past_key_values is None or len(past_key_values.key_cache) < self.config.base_config.num_hidden_layers:
            norm_llm_hidden_states = self.llm.layers[index].input_layernorm(llm_hidden_states)

        ########## Self Attention Begins ##########

        dit_query_states = self.dit.layers[index - self.config.initial_layers].self_attn.q_proj(norm_dit_hidden_states)
        dit_key_states = self.dit.layers[index - self.config.initial_layers].self_attn.k_proj(norm_dit_hidden_states)
        dit_value_states = self.dit.layers[index - self.config.initial_layers].self_attn.v_proj(norm_dit_hidden_states)

        dit_query_states = rearrange(dit_query_states, "b n (h d) -> b h n d", h=self.config.base_config.num_attention_heads)
        dit_key_states = rearrange(dit_key_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)
        dit_value_states = rearrange(dit_value_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)

        if past_key_values is None or len(past_key_values.key_cache) < self.config.base_config.num_hidden_layers:
            llm_query_states = self.llm.layers[index].self_attn.q_proj(norm_llm_hidden_states)
            llm_key_states = self.llm.layers[index].self_attn.k_proj(norm_llm_hidden_states)
            llm_value_states = self.llm.layers[index].self_attn.v_proj(norm_llm_hidden_states)
        
            llm_query_states = rearrange(llm_query_states, "b n (h d) -> b h n d", h=self.config.base_config.num_attention_heads)
            llm_key_states = rearrange(llm_key_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)
            llm_value_states = rearrange(llm_value_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)

        if self.config.qk_norm:
            dit_query_states = self.dit.layers[index - self.config.initial_layers].self_attn.q_norm(dit_query_states)
            dit_key_states = self.dit.layers[index - self.config.initial_layers].self_attn.k_norm(dit_key_states)

        if self.config.attention == "self":
            if past_key_values is None or len(past_key_values.key_cache) < self.config.base_config.num_hidden_layers:
                if self.config.pos_embed == "ape": # RoPE only for LLM, APE for DiT
                    cos, sin = pos_embed
                    llm_query_states, llm_key_states = apply_rotary_pos_emb(llm_query_states, llm_key_states, cos, sin)

                query_states = torch.cat([llm_query_states, dit_query_states], dim=2)
                key_states = torch.cat([llm_key_states, dit_key_states], dim=2)
                value_states = torch.cat([llm_value_states, dit_value_states], dim=2)

                if not self.config.pos_embed == "ape": # RoPE for both LLM and DiT
                    cos, sin = pos_embed
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_values is not None:
                    past_key_values.update(key_states[:, :, :-dit_hidden_states.shape[1]], value_states[:, :, :-dit_hidden_states.shape[1]], index - self.config.initial_layers)
            else:
                if not self.config.pos_embed == "ape": # RoPE for both LLM and DiT
                    cos, sin = pos_embed
                    dit_query_states, dit_key_states = apply_rotary_pos_emb(dit_query_states, dit_key_states, cos, sin)

                query_states = dit_query_states
                key_states = torch.cat([past_key_values.key_cache[index - self.config.initial_layers], dit_key_states], dim=2)
                value_states = torch.cat([past_key_values.value_cache[index - self.config.initial_layers], dit_value_states], dim=2)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, is_causal=False)

            attn_output = rearrange(attn_output, "b h n d -> b n (h d)")

            dit_attn_output = attn_output[:, -dit_hidden_states.shape[1]:]
            dit_attn_output = self.dit.layers[index - self.config.initial_layers].self_attn.o_proj(dit_attn_output)
            if self.config.sandwich_norm:
                dit_attn_output = self.dit.layers[index - self.config.initial_layers].post_attention_layernorm(dit_attn_output)
            dit_hidden_states = dit_hidden_states + dit_gate_msa.unsqueeze(1) * dit_attn_output

            if past_key_values is None or len(past_key_values.key_cache) < self.config.base_config.num_hidden_layers:
                llm_attn_output = attn_output[:, :-dit_hidden_states.shape[1]]
                llm_attn_output = self.llm.layers[index].self_attn.o_proj(llm_attn_output)
                if self.config.base_config.model_type == "gemma2":
                    llm_attn_output = self.llm.layers[index].post_attention_layernorm(llm_attn_output)
                llm_hidden_states = llm_hidden_states + llm_attn_output
        elif self.config.attention == "cross":
            if past_key_values is None or len(past_key_values.key_cache) < self.config.base_config.num_hidden_layers:
                if past_key_values is not None:
                    past_key_values.update(llm_key_states, llm_value_states, index - self.config.initial_layers)

                if self.config.pos_embed == "ape": # RoPE only for LLM, APE for DiT
                    cos, sin = pos_embed
                    rope_llm_query_states, rope_llm_key_states = apply_rotary_pos_emb(llm_query_states, llm_key_states, cos, sin)
                else:
                    position_ids = torch.arange(
                        llm_hidden_states.shape[1], device=llm_hidden_states.device
                    ).unsqueeze(0)
                    llm_pos_embed = self.dit.rotary_emb(llm_hidden_states, position_ids)
                    cos, sin = llm_pos_embed
                    rope_llm_query_states, rope_llm_key_states = apply_rotary_pos_emb(llm_query_states, llm_key_states, cos, sin)

                rope_llm_key_states = repeat_kv(rope_llm_key_states, self.num_key_value_groups)
                llm_key_states = repeat_kv(llm_key_states, self.num_key_value_groups)
                llm_value_states = repeat_kv(llm_value_states, self.num_key_value_groups)

                llm_attn_output = F.scaled_dot_product_attention(
                    rope_llm_query_states,
                    rope_llm_key_states,
                    llm_value_states,
                    attn_mask=attention_mask,
                    is_causal=False
                )

                llm_attn_output = rearrange(llm_attn_output, "b h n d -> b n (h d)")

                llm_attn_output = self.llm.layers[index].self_attn.o_proj(llm_attn_output)

                if self.config.base_config.model_type == "gemma2":
                    llm_attn_output = self.llm.layers[index].post_attention_layernorm(llm_attn_output)
            else:
                llm_key_states = past_key_values.key_cache[index - self.config.initial_layers]
                llm_value_states = past_key_values.value_cache[index - self.config.initial_layers]

            if not self.config.pos_embed == "ape": # RoPE for both LLM and DiT
                cos, sin = pos_embed
                dit_query_states, dit_key_states = apply_rotary_pos_emb(dit_query_states, dit_key_states, cos, sin)

            dit_key_states = repeat_kv(dit_key_states, self.num_key_value_groups)
            dit_value_states = repeat_kv(dit_value_states, self.num_key_value_groups)

            dit_attn_output = F.scaled_dot_product_attention(
                dit_query_states,
                dit_key_states,
                dit_value_states,
                attn_mask=None,
                is_causal=False
            )

            dit_attn_output = rearrange(dit_attn_output, "b h n d -> b n (h d)")

            dit_attn_output = self.dit.layers[index - self.config.initial_layers].self_attn.o_proj(dit_attn_output)

            if self.config.sandwich_norm:
                dit_attn_output = self.dit.layers[index - self.config.initial_layers].post_attention_layernorm(dit_attn_output)

        ########## Self Attention Ends ##########

        ########## Cross Attention Begins ##########

            dit_hidden_states = dit_hidden_states + dit_gate_msa.unsqueeze(1) * dit_attn_output

            dit_cross_attn_output = self.dit.layers[index - self.config.initial_layers].cross_attn(
                dit_hidden_states,
                key_states=llm_key_states.to(dit_hidden_states.dtype),
                value_states=llm_value_states.to(dit_hidden_states.dtype),
                attention_mask=cross_attention_mask
            )

            if self.config.sandwich_norm:
                dit_cross_attn_output = self.dit.layers[index - self.config.initial_layers].post_cross_attention_layernorm(dit_cross_attn_output)

            dit_hidden_states = dit_hidden_states + dit_cross_attn_output

            if past_key_values is None or len(past_key_values.key_cache) < self.config.base_config.num_hidden_layers:
                llm_hidden_states = llm_hidden_states + llm_attn_output

        ########## Cross Attention Ends ##########

        else:
            raise ValueError(f"Unknown attention type: {self.config.attention}")

        ########## Feedforward Begins ##########
        
        if self.config.sandwich_norm:
            norm_dit_hidden_states = self.dit.layers[index - self.config.initial_layers].pre_feedforward_layernorm(dit_hidden_states)
        else:
            norm_dit_hidden_states = self.dit.layers[index - self.config.initial_layers].post_attention_layernorm(dit_hidden_states)
        norm_dit_hidden_states = norm_dit_hidden_states * (1 + dit_scale_mlp[:, None]) + dit_shift_mlp[:, None]
        dit_mlp_output = self.dit.layers[index - self.config.initial_layers].mlp(norm_dit_hidden_states)
        if self.config.sandwich_norm:
            dit_mlp_output = self.dit.layers[index - self.config.initial_layers].post_feedforward_layernorm(dit_mlp_output)
        dit_hidden_states = dit_hidden_states + dit_gate_mlp.unsqueeze(1) * dit_mlp_output

        if past_key_values is None or len(past_key_values.key_cache) < self.config.base_config.num_hidden_layers:
            if self.config.base_config.model_type == "gemma2":
                norm_llm_hidden_states = self.llm.layers[index].pre_feedforward_layernorm(llm_hidden_states)
            else:
                norm_llm_hidden_states = self.llm.layers[index].post_attention_layernorm(llm_hidden_states)
            llm_mlp_output = self.llm.layers[index].mlp(norm_llm_hidden_states)
            if self.config.base_config.model_type == "gemma2":
                llm_mlp_output = self.llm.layers[index].post_feedforward_layernorm(llm_mlp_output)
            llm_hidden_states = llm_hidden_states + llm_mlp_output

        return dit_hidden_states, llm_hidden_states
    
    def prepare_hidden_states(
        self,
        llm_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        dit_hidden_states: torch.FloatTensor,
        llm_attention_mask: torch.LongTensor,
        height: int,
        width: int,
        use_cache: bool,
    ):
        dit_sequence_length = dit_hidden_states.shape[1]
        llm_sequence_length = llm_hidden_states.shape[1]
        
        if not self.config.pos_embed == "ape":
            if self.config.timestep_conditioning == "addition":
                dit_hidden_states += temb.unsqueeze(1)
            
            if self.config.pos_embed == "m-rope" or self.config.pos_embed == "2d-rope":
                h_position_ids = torch.arange(height, device=llm_hidden_states.device).repeat_interleave(width)
                h_position_ids = repeat(h_position_ids, "s -> b s", b=llm_hidden_states.shape[0])
                w_position_ids = torch.arange(width, device=llm_hidden_states.device).repeat(height)
                w_position_ids = repeat(w_position_ids, "s -> b s", b=llm_hidden_states.shape[0])

                if self.config.attention == "self" and not use_cache:
                    padding = (1 - llm_attention_mask).sum(-1, keepdim=True)
                    h_position_ids = h_position_ids + llm_sequence_length - padding # Remove padding tokens
                    w_position_ids = w_position_ids + llm_sequence_length - padding # Remove padding tokens

                if self.config.pos_embed == "m-rope":
                    if self.config.attention == "self" and not use_cache:                        
                        llm_position_ids = repeat(torch.arange(llm_sequence_length, device=llm_hidden_states.device), "s -> b s", b=llm_hidden_states.shape[0])
                        h_position_ids = torch.cat([llm_position_ids, h_position_ids], dim=1)
                        w_position_ids = torch.cat([llm_position_ids, w_position_ids], dim=1)

                    h_cos, h_sin = self.dit.rotary_emb(torch.cat([llm_hidden_states, dit_hidden_states], dim=1), h_position_ids)
                    w_cos, w_sin = self.dit.rotary_emb(torch.cat([llm_hidden_states, dit_hidden_states], dim=1), w_position_ids)
                    cos = torch.stack([h_cos, w_cos], dim=0)
                    cos = torch.cat([m[i % 2] for i, m in enumerate(cos.chunk(4, dim=-1))], dim=-1)
                    sin = torch.stack([h_sin, w_sin], dim=0)
                    sin = torch.cat([m[i % 2] for i, m in enumerate(sin.chunk(4, dim=-1))], dim=-1)

                    pos_embed = (cos, sin)

                elif self.config.pos_embed == "2d-rope":
                    h_cos, h_sin = self.dit.rotary_emb_half(torch.cat([llm_hidden_states, dit_hidden_states], dim=1), h_position_ids)
                    w_cos, w_sin = self.dit.rotary_emb_half(torch.cat([llm_hidden_states, dit_hidden_states], dim=1), w_position_ids)
                    cos = torch.stack([h_cos, w_cos], dim=2)
                    cos = torch.cat([rearrange(m, "b s n d -> b s (n d)") for m in cos.chunk(2, dim=-1)], dim=-1)
                    sin = torch.stack([h_sin, w_sin], dim=2)
                    sin = torch.cat([rearrange(m, "b s n d -> b s (n d)") for m in sin.chunk(2, dim=-1)], dim=-1)

                    if self.config.attention == "self" and not use_cache:
                        llm_position_ids = repeat(torch.arange(llm_sequence_length, device=llm_hidden_states.device), "s -> b s", b=llm_hidden_states.shape[0])
                        llm_cos, llm_sin = self.dit.rotary_emb(llm_hidden_states, llm_position_ids)

                        cos = torch.cat([llm_cos, cos], dim=1)
                        sin = torch.cat([llm_sin, sin], dim=1)

                    pos_embed = (cos, sin)
            else:
                if self.config.attention == "self" or not use_cache:
                    position_ids = torch.arange(
                        llm_sequence_length + dit_sequence_length, device=llm_hidden_states.device
                    ).unsqueeze(0).expand(llm_hidden_states.shape[0], -1).contiguous()
                    position_ids[:, llm_sequence_length:] -= (1 - llm_attention_mask).sum(-1, keepdim=True) # Remove padding tokens
                else:
                    position_ids = torch.arange(
                        dit_sequence_length, device=dit_hidden_states.device
                    ).unsqueeze(0).expand(dit_hidden_states.shape[0], -1).contiguous()

                pos_embed = self.dit.rotary_emb(torch.cat([llm_hidden_states, dit_hidden_states], dim=1), position_ids)
        else:
            if self.config.timestep_conditioning == "addition":
                dit_hidden_states += temb.unsqueeze(1)
                
            position_ids = torch.arange(
                llm_hidden_states.shape[1], device=llm_hidden_states.device
            ).unsqueeze(0)
            pos_embed = self.dit.rotary_emb(llm_hidden_states, position_ids)

        return dit_hidden_states, pos_embed

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        text_modulation_embeds: Optional[torch.FloatTensor] = None,

        use_cache: bool = False,
        past_key_values: Optional[DynamicCache] = None,
    ):
        height, width = hidden_states.shape[-2:]
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        dit_hidden_states = self.dit.patch_embed(hidden_states)
        if past_key_values is None:
            if use_cache:
                past_key_values = DynamicCache()

            llm_hidden_states = self.llm.embed_tokens(input_ids)
            if "gemma" in self.config.base_config.model_type:
                llm_hidden_states *= torch.tensor(self.config.base_config.hidden_size ** 0.5, dtype=self.llm.dtype)
        else:
            llm_hidden_states = torch.zeros(hidden_states.shape[0], past_key_values.get_seq_length(), self.config.base_config.hidden_size, device=hidden_states.device)

        if self.config.timestep_conditioning is not None:
            timestep = self.dit.time_proj(timestep).to(dit_hidden_states.dtype)
            timestep = self.dit.timestep_embedder(timestep)

            if self.config.text_modulation_embeds_dim is not None:
                timestep = timestep + self.dit.condition_embedder(text_modulation_embeds)

            if self.config.timestep_conditioning == "adaln-single":
                temb = self.dit.t_block(timestep)
            else:
                temb = timestep
        else:
            temb = torch.zeros_like(dit_hidden_states[:, 0, :])

        dit_hidden_states, pos_embed = self.prepare_hidden_states(
            llm_hidden_states,
            temb,
            dit_hidden_states,
            attention_mask,
            height,
            width,
            use_cache and past_key_values.get_seq_length() > 0,
        )

        self_attention_mask = update_self_attention_mask(
            attention_mask,
            dit_hidden_states.shape[1] if self.config.attention == "self" else 0,
            use_cache and past_key_values.get_seq_length() > 0,
            llm_hidden_states.device,
            torch.float32 if ACCEL == "xla" else llm_hidden_states.dtype,
        )

        if self.config.attention == "cross":
            cross_attention_mask = update_cross_attention_mask(
                attention_mask,
                dit_hidden_states.shape[1],
                False,
                llm_hidden_states.device,
                torch.float32 if ACCEL == "xla" else llm_hidden_states.dtype,
            )
        else:
            cross_attention_mask = None

        for layer_index in range(self.config.initial_layers):
            if self.gradient_checkpointing and self.training:
                llm_hidden_states = self._gradient_checkpointing_func(
                    self.llm.layers[layer_index].__call__,
                    llm_hidden_states,
                    self_attention_mask[:, :, :llm_hidden_states.shape[1], :llm_hidden_states.shape[1]],
                    repeat(torch.arange(llm_hidden_states.shape[1], device=llm_hidden_states.device), "s -> b s", b=llm_hidden_states.shape[0])
                )[0]

            else:
                llm_hidden_states = self.llm.layers[layer_index](
                    llm_hidden_states,
                    self_attention_mask[:, :, :llm_hidden_states.shape[1], :llm_hidden_states.shape[1]],
                    repeat(torch.arange(llm_hidden_states.shape[1], device=llm_hidden_states.device), "s -> b s", b=llm_hidden_states.shape[0])
                )[0]

        for layer_index in range(self.config.initial_layers, self.config.base_config.num_hidden_layers):
            if self.gradient_checkpointing and self.training:
                if self.config.shared_attention_layers == "all" or layer_index in self.config.shared_attention_layers:
                    dit_hidden_states, llm_hidden_states = (
                        self._gradient_checkpointing_func(
                            self.shared_transformer_layer,
                            layer_index,
                            dit_hidden_states,
                            llm_hidden_states,
                            temb,
                            pos_embed,
                            self_attention_mask,
                            cross_attention_mask,
                            past_key_values,
                        )
                    )
                else:
                    dit_hidden_states = self._gradient_checkpointing_func(
                        self.dit.layers[layer_index].__call__,
                        dit_hidden_states,
                        temb,
                        None if pos_embed is None else (pos_embed[0][:, llm_hidden_states.shape[1]:], pos_embed[1][:, llm_hidden_states.shape[1]:]),
                    )
                    llm_hidden_states = self._gradient_checkpointing_func(
                        self.llm.layers[layer_index].__call__,
                        llm_hidden_states,
                        self_attention_mask[:, :, :llm_hidden_states.shape[1], :llm_hidden_states.shape[1]],
                        repeat(torch.arange(llm_hidden_states.shape[1], device=llm_hidden_states.device), "s -> b s", b=llm_hidden_states.shape[0])
                    )[0]

            else:
                if self.config.shared_attention_layers == "all" or layer_index in self.config.shared_attention_layers:
                    dit_hidden_states, llm_hidden_states = self.shared_transformer_layer(
                        layer_index,
                        dit_hidden_states,
                        llm_hidden_states,
                        temb,
                        pos_embed,
                        self_attention_mask,
                        cross_attention_mask,
                        past_key_values,
                    )
                else:
                    dit_hidden_states = self.dit.layers[layer_index](
                        dit_hidden_states,
                        temb,
                        None if pos_embed is None else (pos_embed[0][:, llm_hidden_states.shape[1]:], pos_embed[1][:, llm_hidden_states.shape[1]:]),
                    )
                    llm_hidden_states = self.llm.layers[layer_index](
                        llm_hidden_states,
                        self_attention_mask[:, :, :llm_hidden_states.shape[1], :llm_hidden_states.shape[1]],
                        repeat(torch.arange(llm_hidden_states.shape[1], device=llm_hidden_states.device), "s -> b s", b=llm_hidden_states.shape[0])
                    )[0]

        for layer_index in range(self.config.base_config.num_hidden_layers - self.config.initial_layers, self.config.dit_num_hidden_layers):
            if self.gradient_checkpointing and self.training:
                dit_hidden_states = self._gradient_checkpointing_func(
                    self.dit.layers[layer_index].__call__,
                    dit_hidden_states,
                    temb,
                    None if pos_embed is None else (pos_embed[0][:, llm_hidden_states.shape[1]:], pos_embed[1][:, llm_hidden_states.shape[1]:]),
                )

            else:
                dit_hidden_states = self.dit.layers[layer_index](
                    dit_hidden_states,
                    temb,
                    None if pos_embed is None else (pos_embed[0][:, llm_hidden_states.shape[1]:], pos_embed[1][:, llm_hidden_states.shape[1]:]),
                )

        if self.config.timestep_conditioning == "adaln-zero":
            dit_hidden_states = self.dit.norm_out(dit_hidden_states, temb)
        else:
            dit_hidden_states = self.dit.norm_out(dit_hidden_states)
        if self.config.timestep_conditioning == "adaln-single":
            shift, scale = (self.dit.scale_shift_table + repeat(timestep, "b d -> b (2 d)")).chunk(2, dim=1)
            dit_hidden_states = dit_hidden_states * (1 + scale[:, None]) + shift[:, None]
        dit_hidden_states = self.dit.proj_out(dit_hidden_states)

        output = (
            unpatchify(dit_hidden_states, height, width, patch_size),
            past_key_values,
        )

        return output
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = { "use_reentrant": True }

        gradient_checkpointing_func = functools.partial(torch.utils.checkpoint.checkpoint, **gradient_checkpointing_kwargs)
        
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)

    def trainable_parameters(self): # * Make SPMD CheckpointManager happy
        return (
            [param for param in self.dit.parameters() if param.requires_grad]
            + [param for param in self.llm.embed_tokens.parameters() if param.requires_grad]
            + [param for param in self.llm.layers[:-1].parameters() if param.requires_grad]
            + [param for name, param in self.llm.layers[-1].named_parameters()
                if not "o_proj" in name
                and not "post_attention_layernorm" in name
                and not "pre_feedforward_layernorm" in name
                and not "post_feedforward_layernorm" in name
                and not "mlp" in name
                and param.requires_grad
            ]
        )


def build_fusedit(hparams):
    base_config = AutoConfig.from_pretrained(hparams.model.base)
    load_model = get_llm(hparams.model.base, base_config)
    base_config = load_model.config

    assert hparams.model.attention in ["self", "cross"]
    assert hparams.model.pos_embed in ["ape", "1d-rope", "2d-rope", "m-rope"]
    assert hparams.model.timestep_conditioning in [None, "adaln-zero", "adaln-single", "addition"]

    if not hasattr(base_config, "head_dim"):
        base_config.head_dim = base_config.hidden_size // base_config.num_attention_heads
    base_config.num_hidden_layers = min(
        base_config.num_hidden_layers,
        max(hparams.model.shared_attention_layers) + 1 if hparams.model.shared_attention_layers != "all" else base_config.num_hidden_layers,
    )

    config = FuseDiTConfig(
        attention=hparams.model.attention,
        base_config=base_config,
        dit_hidden_size=hparams.model.dit_hidden_size if hasattr(hparams.model, "dit_hidden_size") else base_config.hidden_size,
        dit_num_hidden_layers=hparams.model.dit_num_hidden_layers,
        initial_layers=hparams.model.initial_layers if hasattr(hparams.model, "initial_layers") else 0,
        patch_size=hparams.model.patch_size,
        pos_embed=hparams.model.pos_embed,
        qk_norm=hparams.model.qk_norm,
        sandwich_norm=hparams.model.sandwich_norm,
        shared_attention_layers=hparams.model.shared_attention_layers,
        text_hidden_size=base_config.hidden_size,
        text_modulation_embeds_dim=hparams.model.text_modulation_embeds_dim if hasattr(hparams.model, "text_modulation_embeds_dim") else None,
        timestep_conditioning=hparams.model.timestep_conditioning,
    )
    transformer = FuseDiT(config)

    transformer.llm.load_state_dict(load_model.state_dict(), strict=False)
    del load_model

    transformer.dit.requires_grad_(hparams.trainer.train_dit)
    transformer.llm.requires_grad_(hparams.trainer.train_llm)

    return transformer


def build_dit(hparams):
    """构建标准 DiT 模型"""
    base_config = AutoConfig.from_pretrained(hparams.model.base)
    load_model = get_llm(hparams.model.base, base_config)
    base_config = load_model.config
    del load_model

    assert hparams.model.pos_embed in ["ape", "1d-rope", "2d-rope", "m-rope"]
    assert hparams.model.timestep_conditioning in [None, "adaln-zero", "adaln-single", "addition"]

    if not hasattr(base_config, "head_dim"):
        base_config.head_dim = base_config.hidden_size // base_config.num_attention_heads

    config = DiTConfig(
        attention=hparams.model.attention if hasattr(hparams.model, "attention") else "self",
        base_config=base_config,
        dit_hidden_size=hparams.model.dit_hidden_size if hasattr(hparams.model, "dit_hidden_size") else base_config.hidden_size,
        dit_num_hidden_layers=hparams.model.dit_num_hidden_layers,
        patch_size=hparams.model.patch_size,
        pos_embed=hparams.model.pos_embed,
        qk_norm=hparams.model.qk_norm if hasattr(hparams.model, "qk_norm") else False,
        text_hidden_size=base_config.hidden_size,
        text_hidden_states_index=hparams.model.text_hidden_states_index if hasattr(hparams.model, "text_hidden_states_index") else -1,
        text_modulation_embeds_dim=hparams.model.text_modulation_embeds_dim if hasattr(hparams.model, "text_modulation_embeds_dim") else None,
        timestep_conditioning=hparams.model.timestep_conditioning,
    )
    
    transformer = DiT(config)
    transformer.requires_grad_(hparams.trainer.train_dit)
    
    return transformer


def build_adafusedit(hparams):
    """构建 AdaFuseDiT 模型，支持多层文本特征自适应融合"""
    base_config = AutoConfig.from_pretrained(hparams.model.base)
    load_model = get_llm(hparams.model.base, base_config)
    base_config = load_model.config
    del load_model

    assert hparams.model.pos_embed in ["ape", "1d-rope", "2d-rope", "m-rope"]
    assert hparams.model.timestep_conditioning in [None, "adaln-zero", "adaln-single", "addition"]

    if not hasattr(base_config, "head_dim"):
        base_config.head_dim = base_config.hidden_size // base_config.num_attention_heads

    config = DiTConfig(
        attention=hparams.model.attention if hasattr(hparams.model, "attention") else "self",
        base_config=base_config,
        dit_hidden_size=hparams.model.dit_hidden_size if hasattr(hparams.model, "dit_hidden_size") else base_config.hidden_size,
        dit_num_hidden_layers=hparams.model.dit_num_hidden_layers,
        patch_size=hparams.model.patch_size,
        pos_embed=hparams.model.pos_embed,
        qk_norm=hparams.model.qk_norm if hasattr(hparams.model, "qk_norm") else False,
        text_hidden_size=base_config.hidden_size,
        text_hidden_states_index=hparams.model.text_hidden_states_index if hasattr(hparams.model, "text_hidden_states_index") else -1,
        text_modulation_embeds_dim=hparams.model.text_modulation_embeds_dim if hasattr(hparams.model, "text_modulation_embeds_dim") else None,
        timestep_conditioning=hparams.model.timestep_conditioning,
        # AdaFuseDiT 特有配置
        text_hidden_states_num=hparams.model.text_hidden_states_num if hasattr(hparams.model, "text_hidden_states_num") else 1,
        use_timestep_adaptive_fusion=hparams.model.use_timestep_adaptive_fusion if hasattr(hparams.model, "use_timestep_adaptive_fusion") else False,
        use_layer_wise_fusion=hparams.model.use_layer_wise_fusion if hasattr(hparams.model, "use_layer_wise_fusion") else False,
        adaptive_fusion_time_embed_dim=hparams.model.adaptive_fusion_time_embed_dim if hasattr(hparams.model, "adaptive_fusion_time_embed_dim") else 128,
    )
    
    transformer = AdaFuseDiT(config)
    transformer.requires_grad_(hparams.trainer.train_dit)
    
    return transformer


def build_model(hparams):
    if hparams.model.name == "DiT":
        return build_dit(hparams)
    elif hparams.model.name == "AdaFuseDiT":
        return build_adafusedit(hparams)
    elif hparams.model.name == "FuseDiT":
        return build_fusedit(hparams)