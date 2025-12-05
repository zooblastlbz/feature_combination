#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from transformers import AutoProcessor
import argparse
import logging
import math
import os
import random
import shutil
import time  # 新增：用于计算训练速度
from contextlib import nullcontext
from pathlib import Path
from typing import List, Dict, Any

import accelerate
import datasets

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, DatasetDict, load_from_disk, concatenate_datasets
from datasets import Dataset as HFDataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, Gemma2Model, T5EncoderModel, T5Tokenizer
from transformers.utils.generic import ContextManagers

import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, Lumina2Pipeline, Lumina2Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_loss_weighting_for_sd3, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.image_processor import VaeImageProcessor
# 新增：强制使用 Lumina2 自定义注意力处理器
from diffusers.models.transformers.transformer_lumina2 import Lumina2AttnProcessor2_0
from diffusers.models.attention_processor import Attention as DiffusersAttention

os.environ.setdefault('NCCL_TIMEOUT', '1800')  
if is_wandb_available():
    import wandb

class AvgNormLayer(torch.nn.Module):
    """
    对文本编码器的所有隐藏层进行LayerNorm后加权求和的模块
    使用nn.functional.layer_norm而不是LayerNorm层
    """
    def __init__(self, num_layers: int, hidden_size: int, eps: float = 1e-5,trainable_weights=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.eps = eps
        self.trainable_weights=trainable_weights
        # 可学习的权重参数，使用正态分布初始化
        self.layer_weights = torch.nn.Parameter(
            torch.empty(num_layers).normal_(0, 0.1)
        )
        if not self.trainable_weights:
            self.layer_weights = torch.nn.Parameter(
    torch.zeros(num_layers)  # 初始化为全0
)
            self.layer_weights.requires_grad_(False)
        else:
            self.layer_weights = torch.nn.Parameter(
            torch.empty(num_layers).normal_(0, 0.1)
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hidden_states: List of hidden states from text encoder, 
                         each of shape (batch_size, sequence_length, hidden_size)
        
        Returns:
            weighted_hidden: Weighted sum of normalized hidden states, 
                           shape (batch_size, sequence_length, hidden_size)
        """
        # 确保hidden_states是列表
        if isinstance(hidden_states, tuple):
            hidden_states = list(hidden_states)
        
        # 检查隐藏状态数量
        available_layers = len(hidden_states)
        if available_layers != self.num_layers:
            logger.warning(
                f"AvgNormLayer期望{self.num_layers}层，但得到{available_layers}层。"
                f"将进行调整以适应。"
            )
            
            if available_layers < self.num_layers:
                # 如果可用层数少于预期，重复使用最后一层
                last_layer = hidden_states[-1]
                hidden_states = hidden_states + [last_layer] * (self.num_layers - available_layers)
            else:
                # 如果可用层数多于预期，截断
                hidden_states = hidden_states[-self.num_layers:]
        
        # 对每一层的隐藏状态应用LayerNorm（使用functional）
        normalized_states = []
        for hidden_state in hidden_states:
            # 确保hidden_state是正确形状的张量
            if hidden_state.dim() != 3:
                logger.warning(f"隐藏状态形状异常: {hidden_state.shape}，尝试重塑")
                # 尝试重塑为(batch_size, seq_len, hidden_size)
                if hidden_state.dim() == 2:
                    hidden_state = hidden_state.unsqueeze(1)
            
            # 使用functional layer_norm，不包含可训练参数
            norm_state = F.layer_norm(
                hidden_state, 
                [self.hidden_size], 
                weight=None, 
                bias=None, 
                eps=self.eps
            )
            normalized_states.append(norm_state)
        
        # 应用softmax到权重，确保权重和为1
        weights = torch.softmax(self.layer_weights, dim=0)
        
        # 加权求和
        weighted_hidden = torch.zeros_like(normalized_states[0])
        for i, norm_state in enumerate(normalized_states):
            weighted_hidden += weights[i] * norm_state
        
        return weighted_hidden

class Lumina2TransformerWithAvgNorm(Lumina2Transformer2DModel):
    def __init__(self, *args, **kwargs):
        # 提取AvgNormLayer相关参数
        use_avg_norm_layer = kwargs.pop('use_avg_norm_layer', False)
        avg_norm_num_layers = kwargs.pop('avg_norm_num_layers', None)
        avg_norm_hidden_size = kwargs.pop('avg_norm_hidden_size', None)
        avg_norm_eps = kwargs.pop('avg_norm_eps', 1e-5)
        avg_norm_trainable_weights = kwargs.pop('avg_norm_trainable_weights', True)  # 新增参数，默认可训练
        
        super().__init__(*args, **kwargs)
        
        # 添加AvgNormLayer
        self.use_avg_norm_layer = use_avg_norm_layer
        if use_avg_norm_layer and avg_norm_num_layers and avg_norm_hidden_size:
            self.avg_norm_layer = AvgNormLayer(
                num_layers=avg_norm_num_layers,
                hidden_size=avg_norm_hidden_size,
                eps=avg_norm_eps,
                trainable_weights=avg_norm_trainable_weights  # 传递新参数
            )  

    
    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask=None, return_dict=True):
        # 如果使用AvgNormLayer，处理encoder_hidden_states
        if self.use_avg_norm_layer and hasattr(self, 'avg_norm_layer'):
            # 这里encoder_hidden_states应该是一个包含所有层的列表
            # 应用AvgNormLayer来聚合所有层
            encoder_hidden_states = self.avg_norm_layer(encoder_hidden_states)
        
        # 调用父类的forward方法
        return super().forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict
        )
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
#check_min_version("0.35.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}

# 在你的脚本顶部或一个单独的文件中定义这个类
import torch

class SafeDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, preprocess_fn):
        self.hf_dataset = hf_dataset
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        try:
            # 获取原始数据样本（这是一个字典）
            example = self.hf_dataset[idx]
            # 使用你的预处理逻辑（现在它一次只处理一个样本）
            # 注意：你需要稍微修改 PreprocessTrain 的 __call__ 来处理单个样本而非批次
            processed_example = self.preprocess_fn(example)
            return processed_example
        except Exception as e:
            # 如果任何环节出错，打印错误并返回 None
            logger.warning(f"跳过索引 {idx} 的数据，错误: {e}")
            return None

# 你需要一个处理单个样本的函数
def create_preprocess_fn(tokenizer, args, train_transforms):
    # 这个函数返回一个闭包，用于处理单个样本
    def preprocess_single(example):
        # 这里的逻辑是从你原来的 PreprocessTrain.__call__ 中提取的，但针对单个样本
        # 1. 安全加载图片
        image = Image.open(example["image"]).convert("RGB") # 简化版，你可以用你原来的_load_image_safely

        # 2. 图片变换
        pixel_values = train_transforms(image)

        # 3. 处理标题
        caption = example["text"]
        if isinstance(caption, list):
            caption = random.choice(caption)
        
        # 4. Prompt Drop
        if random.random() < args.prompt_drop_probability:
            caption = ""

        # 5. Tokenize
        inputs = tokenizer(
            caption,
            max_length=min(args.caption_max_length, tokenizer.model_max_length),
            padding="max_length", # 对单个样本使用 max_length 填充
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": inputs.input_ids.squeeze(), # 移除批次维度
            "attention_mask": inputs.attention_mask.squeeze(),
        }
    return preprocess_single

def save_model_card(
    args,
    repo_id: str,
    images = None,
    repo_folder = None,
):
    if images is None:
        images = []
    if repo_folder is None:
        repo_folder = args.output_dir
        
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    model_description = f"""
# Lumina2 finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: 
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import Lumina2Pipeline
import torch

pipeline = Lumina2Pipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{args.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

        if wandb_run_url is not None:
            wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_description += wandb_info

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model="Lumina2 (trained from scratch)",
        model_description=model_description,
        inference=True,
    )

    tags = ["lumina2", "lumina", "text-to-image", "diffusers", "diffusers-training"]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(vae, text_encoder, tokenizer, transformer, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    # Create pipeline with our custom models
    pipeline = Lumina2Pipeline(
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        transformer=accelerator.unwrap_model(transformer),
        scheduler=FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=False,
        )
    )
    pipeline = pipeline.to(accelerator.device)


    # 禁用 xFormers，避免与 Lumina2 自定义注意力冲突
    pipeline = _maybe_enable_xformers_for_pipeline(pipeline)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        # 当使用 DeepSpeed 时禁用手动 autocast
        ds_active = getattr(accelerator.state, "deepspeed_plugin", None) is not None
        use_autocast = (accelerator.device.type != "cpu") and (accelerator.mixed_precision in ["fp16", "bf16"]) and (not ds_active)
        autocast_ctx = torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype) if use_autocast else nullcontext()
        with autocast_ctx:
            image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a Lumina2 training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data or a JSON file with image-text pairs. "
            "For folders, contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. "
            "For JSON files, each line should contain 'image' and 'text' fields. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lumina2-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution. For Lumina2, common resolutions are 256, 512, 1024."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://huggingface.co/papers/2303.09556.",
    )
    parser.add_argument(
        "--dream_training",
        action="store_true",
        help=(
            "Use the DREAM training method, which makes training more efficient and accurate at the "
            "expense of doing an extra forward pass. See: https://huggingface.co/papers/2312.00210"
        ),
    )
    parser.add_argument(
        "--dream_detail_preservation",
        type=float,
        default=1.0,
        help="Dream detail preservation factor p (should be greater than 0; default=1.0, as suggested in the paper)",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use faster foreach implementation of EMAModel.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="bilinear",
        help="The interpolation mode to use when resizing images. Choose between 'bilinear', 'bicubic', 'nearest', etc.",
    )
    # 新增：限制 caption 最大长度，避免长序列 OOM
    parser.add_argument(
        "--caption_max_length",
        type=int,
        default=4096,
        help="Max token length for captions to avoid OOM; tokenizer will truncate to this length.",
    )
    parser.add_argument(
        "--dataset_snapshot_dir",
        type=str,
        default=None,
        help="If set, save a HF Datasets snapshot (save_to_disk) to this directory when it does not exist.",
    )
    
    # Model configuration arguments for training from scratch
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=16,
        help="Number of attention heads in the transformer.",
    )
    parser.add_argument(
        "--attention_head_dim",
        type=int,
        default=72,
        help="Dimension of each attention head.",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=4,
        help="Number of input channels (latent channels from VAE).",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=8,
        help="Number of output channels.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=28,
        help="Number of transformer layers.",
    )
    parser.add_argument(
        "--cross_attention_dim",
        type=int,
        default=4096,
        help="Dimension of cross attention (should match text encoder hidden size).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=32,
        help="Sample size for the transformer (resolution // 8 for VAE latents).",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=2,
        help="Patch size for the transformer.",
    )
    parser.add_argument(
        "--num_embeds_ada_norm",
        type=int,
        default=1000,
        help="Number of embeddings for AdaLayerNorm.",
    )
    parser.add_argument(
        "--caption_channels",
        type=int,
        default=4096,
        help="Number of channels for caption projection (should match text encoder hidden size).",
    )
    parser.add_argument(
        "--text_encoder_name",
        type=str,
        default="google/gemma-2-2b",
        help="Text encoder model name or path.",
    )
    parser.add_argument(
        "--vae_name",
        type=str,
        default="stabilityai/sd-vae-ft-ema",
        help="VAE model name or path.",
    )
    parser.add_argument(
        "--cap_feat_dim_override",
        type=int,
        default=None,
        help="Override the caption feature dimension (cap_feat_dim). If not set, will use text encoder's hidden_size.",
    )
    # 新增：Prompt Drop 相关参数
    parser.add_argument(
        "--prompt_drop_probability",
        type=float,
        default=0.1,
        help="Probability of randomly dropping text prompts during training for unconditional generation capability.",
    )
    
    parser.add_argument(
        "--stage_two",
        action="store_true",
        default=False,
    )
    
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--use_avg_norm_layer",
        action="store_true",
        help="Whether to use average norm layer for text encoder hidden states.",
)   
    parser.add_argument(
        "--avg_norm_trainable_weights",
        action="store_true", 
        default=True,  # 默认可训练
        help="Whether the weights in AvgNormLayer are trainable (default: True). Use --no-avg_norm_trainable_weights to disable.",
    )
    parser.add_argument(
        "--no_avg_norm_trainable_weights",
        action="store_false",
        dest="avg_norm_trainable_weights",  # 这个参数会覆盖上面的默认值
        help="Make weights in AvgNormLayer non-trainable.",
    )
    parser.add_argument(
        "--feature_layer",
        default=None,
        type=int,
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


# 顶层可序列化的预处理与 collate 定义，避免 DataLoader 多进程 pickle 失败
class PreprocessTrain:
    def __init__(self, image_column: str, caption_column: str, tokenizer, caption_max_length: int, train_transforms, prompt_drop_probability: float = 0.1):
        self.image_column = image_column
        self.caption_column = caption_column
        self.tokenizer = tokenizer
        self.caption_max_length = caption_max_length
        self.train_transforms = train_transforms
        # 统计失败次数
        self.failed_count = 0
        self.total_count = 0
        # Prompt drop 配置
        self.prompt_drop_probability = prompt_drop_probability

    def _create_fallback_image(self, target_size=(256, 256)):
        """创建默认的RGB图片作为fallback"""
        try:
            # 创建一个简单的渐变图片作为默认图片
            fallback_image = Image.new('RGB', target_size, color=(128, 128, 128))
            return fallback_image
        except Exception:
            # 如果连创建默认图片都失败，创建纯色图片
            return Image.new('RGB', (256, 256), color=(0, 0, 0))

    def _load_image_safely(self, image_item, index=None):
        """安全地加载图片，包含错误处理"""
        try:
            if isinstance(image_item, str):
                # 检查文件是否存在
                if not os.path.exists(image_item):
                    logger.warning(f"图片文件不存在: {image_item}")
                    return self._create_fallback_image()
                
                # 检查文件大小
                file_size = os.path.getsize(image_item)
                if file_size == 0:
                    logger.warning(f"图片文件为空: {image_item}")
                    return self._create_fallback_image()
                
                # 尝试打开图片
                with Image.open(image_item) as img:
                    # 检查图片模式
                    if img.mode not in ['RGB', 'RGBA', 'L']:
                        logger.warning(f"不支持的图片模式 {img.mode}: {image_item}")
                        return self._create_fallback_image()
                    
                    # 检查图片尺寸
                    if img.size[0] < 32 or img.size[1] < 32:
                        logger.warning(f"图片尺寸过小 {img.size}: {image_item}")
                        return self._create_fallback_image()
                    
                    # 转换为RGB
                    image = img.convert("RGB")
                    
                    # 验证图片数据
                    try:
                        # 尝试获取像素数据来验证图片完整性
                        image.load()
                        return image
                    except Exception as e:
                        logger.warning(f"图片数据损坏: {image_item}, 错误: {e}")
                        return self._create_fallback_image()
                        
            else:
                # PIL Image对象
                if hasattr(image_item, 'convert'):
                    try:
                        image = image_item.convert("RGB")
                        image.load()  # 验证图片数据
                        return image
                    except Exception as e:
                        logger.warning(f"PIL图片对象处理失败: {e}")
                        return self._create_fallback_image()
                else:
                    logger.warning(f"不支持的图片对象类型: {type(image_item)}")
                    return self._create_fallback_image()
                    
        except FileNotFoundError:
            logger.warning(f"图片文件未找到: {image_item}")
            self.failed_count += 1
            return self._create_fallback_image()
        except PermissionError:
            logger.warning(f"无权限访问图片文件: {image_item}")
            self.failed_count += 1
            return self._create_fallback_image()
        except OSError as e:
            logger.warning(f"图片文件系统错误: {image_item}, 错误: {e}")
            self.failed_count += 1
            return self._create_fallback_image()
        except Exception as e:
            logger.warning(f"图片加载未知错误: {image_item}, 错误: {e}")
            self.failed_count += 1
            return self._create_fallback_image()

    def _apply_prompt_drop(self, captions: List[str]) -> List[str]:
        """
        应用prompt drop：以一定概率将文本提示词置为空
        """
        if self.prompt_drop_probability <= 0:
            return captions
        
        dropped_captions = []
        for caption in captions:
            # 以prompt_drop_probability的概率将caption置为空
            if random.random() < self.prompt_drop_probability:
                dropped_captions.append("")  # 空字符串用于无条件生成
            else:
                dropped_captions.append(caption)
        
        return dropped_captions

    def __call__(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        images = []
        valid_indices = []  # 记录有效样本的索引
        
        # 处理图片
        for i, image_item in enumerate(examples[self.image_column]):
            self.total_count += 1
            image = self._load_image_safely(image_item, index=i)
            
            if image is not None:
                try:
                    # 尝试应用图片变换
                    transformed_image = self.train_transforms(image)
                    images.append(transformed_image)
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"图片变换失败，索引 {i}: {e}")
                    # 尝试用默认图片进行变换
                    try:
                        fallback_image = self._create_fallback_image()
                        transformed_image = self.train_transforms(fallback_image)
                        images.append(transformed_image)
                        valid_indices.append(i)
                        self.failed_count += 1
                    except Exception as e2:
                        logger.error(f"连默认图片变换都失败，跳过样本 {i}: {e2}")
                        continue
            else:
                logger.warning(f"图片加载完全失败，跳过样本 {i}")
                continue

        # 如果没有有效的图片，创建一个默认图片避免完全失败
        if len(images) == 0:
            logger.warning("批次中所有图片都加载失败，使用默认图片")
            fallback_image = self._create_fallback_image()
            try:
                transformed_image = self.train_transforms(fallback_image)
                images.append(transformed_image)
                valid_indices.append(0)
            except Exception as e:
                logger.error(f"默认图片变换失败: {e}")
                # 创建最简单的tensor
                import torch
                images.append(torch.zeros(3, 256, 256))
                valid_indices.append(0)

        # 只保留有效样本的caption
        captions: List[str] = []
        original_captions = examples[self.caption_column]
        
        for idx in valid_indices:
            if idx < len(original_captions):
                caption = original_captions[idx]
            else:
                # 如果索引超出范围，使用第一个caption或默认caption
                caption = original_captions[0] if original_captions else "A default image"
            
            try:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    if len(caption) > 0:
                        captions.append(random.choice(caption))
                    else:
                        captions.append("A default image")
                else:
                    logger.warning(f"不支持的caption类型: {type(caption)}")
                    captions.append("A default image")
            except Exception as e:
                logger.warning(f"Caption处理失败: {e}")
                captions.append("A default image")

        # 确保images和captions数量匹配
        min_length = min(len(images), len(captions))
        if min_length == 0:
            # 紧急情况：创建最基本的样本
            logger.error("无法创建有效的训练样本，使用紧急默认值")
            import torch
            images = [torch.zeros(3, 256, 256)]
            captions = ["Emergency default image"]

        images = images[:min_length]
        captions = captions[:min_length]

        # 应用prompt drop
        captions = self._apply_prompt_drop(captions)

        # Token化captions
        try:
            inputs = self.tokenizer(
                captions,
                max_length=min(self.caption_max_length, self.tokenizer.model_max_length),
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )
        except Exception as e:
            logger.error(f"Tokenization失败: {e}")
            # 使用默认的tokenization
            try:
                inputs = self.tokenizer(
                    ["A default image"] * len(captions),
                    max_length=min(self.caption_max_length, self.tokenizer.model_max_length),
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                )
            except Exception as e2:
                logger.error(f"默认tokenization也失败: {e2}")
                # 创建最基本的tokens
                import torch
                inputs = {
                    'input_ids': torch.zeros(len(captions), 10, dtype=torch.long),
                    'attention_mask': torch.ones(len(captions), 10, dtype=torch.long)
                }

        # 记录失败统计
        if self.total_count > 0 and self.total_count % 1000 == 0:
            failure_rate = (self.failed_count / self.total_count) * 100
            logger.info(f"图片加载统计: 总数={self.total_count}, 失败={self.failed_count}, 失败率={failure_rate:.2f}%")

        examples["pixel_values"] = images
        examples["input_ids"] = inputs.input_ids
        examples["attention_mask"] = inputs.attention_mask
        return examples



def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    # 1. 过滤掉所有处理失败的样本 (None)
    original_size = len(examples)
    examples = [e for e in examples if e is not None]
    
    # 2. 如果过滤后整个批次都为空，返回 None 或一个虚拟批次
    if not examples:
        logger.warning(f"整个批次 ({original_size}个样本) 都无法处理，跳过此批次。")
        # 返回 None 会让训练循环跳过这个批次，但需要训练循环能处理 None
        return None 

    # 3. 如果批次有效，正常进行堆叠
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format)
    input_ids = torch.stack([example["input_ids"] for example in examples])
    attention_mask = torch.stack([example["attention_mask"] for example in examples])
    
    return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}

def load_text_encoder_and_tokenizer(text_encoder_name):
    """
    智能加载文本编码器和对应的tokenizer，支持多种模型类型
    """
    try:
        # 首先尝试加载tokenizer来判断模型类型
        if "t5" in text_encoder_name.lower():
            # T5模型
            tokenizer = T5Tokenizer.from_pretrained(text_encoder_name)
            text_encoder = T5EncoderModel.from_pretrained(text_encoder_name)
            logger.info(f"成功加载T5文本编码器: {text_encoder_name}")
        else:
            # 其他模型，先尝试AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
            # 根据tokenizer类型选择对应的模型
            if hasattr(tokenizer, 'model_type'):
                if tokenizer.model_type == 't5':
                    text_encoder = T5EncoderModel.from_pretrained(text_encoder_name)
                    logger.info(f"通过model_type检测到T5，加载T5编码器: {text_encoder_name}")
                elif tokenizer.model_type == 'gemma2':
                    text_encoder = Gemma2Model.from_pretrained(text_encoder_name)
                    logger.info(f"加载Gemma2文本编码器: {text_encoder_name}")
                else:
                    # 尝试AutoModel
                    text_encoder = AutoModel.from_pretrained(text_encoder_name)
                    logger.info(f"使用AutoModel加载文本编码器: {text_encoder_name}")
            else:
                # 没有model_type，尝试AutoModel
                text_encoder = AutoModel.from_pretrained(text_encoder_name)
                logger.info(f"使用AutoModel加载文本编码器: {text_encoder_name}")
    except Exception as e:
        logger.warning(f"智能加载失败: {e}")
        # 回退策略：按顺序尝试不同的加载方式
        try:
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
            text_encoder = AutoModel.from_pretrained(text_encoder_name)
            logger.info(f"回退策略成功，使用AutoModel: {text_encoder_name}")
        except Exception as e2:
            try:
                tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
                text_encoder = Gemma2Model.from_pretrained(text_encoder_name)
                logger.info(f"回退策略成功，使用Gemma2Model: {text_encoder_name}")
            except Exception as e3:
                try:
                    tokenizer = T5Tokenizer.from_pretrained(text_encoder_name)
                    text_encoder = T5EncoderModel.from_pretrained(text_encoder_name)
                    logger.info(f"回退策略成功，使用T5模型: {text_encoder_name}")
                except Exception as e4:
                    raise ValueError(f"无法加载文本编码器 {text_encoder_name}，尝试了所有方法都失败: {e4}")
    
    return text_encoder, tokenizer


def _maybe_enable_xformers_for_pipeline(pipeline):
    """
    为pipeline启用xformers（如果可用），但对Lumina2保持兼容性
    """
    try:
        if is_xformers_available():
            # 检查是否为Lumina2模型，如果是则跳过xformers
            if hasattr(pipeline, 'transformer') and 'Lumina2' in str(type(pipeline.transformer)):
                logger.info("检测到Lumina2模型，跳过xformers以避免兼容性问题")
                return pipeline
            else:
                pipeline.enable_xformers_memory_efficient_attention()
        else:
            logger.info("xformers不可用，使用默认注意力实现")
    except Exception as e:
        logger.warning(f"启用xformers失败: {e}，使用默认注意力实现")
    
    return pipeline


def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)



    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    '''
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    '''
    

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Create scheduler, tokenizer and models.
    # Use FlowMatchEulerDiscreteScheduler for Lumina2
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    
    text_encoder, tokenizer = load_text_encoder_and_tokenizer(args.text_encoder_name)

    # Get text encoder output dimension for cap_feat_dim
    text_encoder_dim = text_encoder.config.hidden_size
    logger.info(f"Text encoder output dimension: {text_encoder_dim}")
    if args.use_avg_norm_layer:
    # 获取文本编码器的层数
        if hasattr(text_encoder.config, 'num_hidden_layers'):
            num_text_layers = text_encoder.config.num_hidden_layers
        else:
            # 对于某些模型，可能需要不同的属性名
            num_text_layers = getattr(text_encoder.config, 'n_layer', 
                                    getattr(text_encoder.config, 'num_layers', 12))
        
    # Override cross_attention_dim with text encoder dimension if not explicitly set
    if args.cap_feat_dim_override is None:
        args.cross_attention_dim = text_encoder_dim
        logger.info(f"Setting cap_feat_dim to text encoder dimension: {text_encoder_dim}")
    else:
        args.cross_attention_dim = args.cap_feat_dim_override
        logger.info(f"Using custom cap_feat_dim: {args.cap_feat_dim_override}")

    # Load VAE
    vae = AutoencoderKL.from_pretrained(args.vae_name)
    logger.info(f"成功加载VAE: {args.vae_name}")

    # Create transformer from config or load from pretrained
    if args.pretrained_model_name_or_path and not args.stage_two and not args.from_scratch:
        transformer = Lumina2Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=args.revision,
            variant=args.variant,
        )
        #transformer=Lumina2Transformer2DModel(os.path.join(args.pretrained_model_name_or_path,"transformer","config.json"))
        # Check if the loaded transformer's cap_feat_dim matches the text encoder dimension
        current_cap_feat_dim = transformer.config.cap_feat_dim
        required_cap_feat_dim = args.cross_attention_dim
        
        if current_cap_feat_dim != required_cap_feat_dim:
            logger.warning(
                f"Transformer's cap_feat_dim ({current_cap_feat_dim}) doesn't match text encoder dimension ({required_cap_feat_dim}). "
                f"Reinitializing caption_embedder..."
            )
            
            # Reinitialize only the caption embedder part, keeping timestep_embedder unchanged
            from diffusers.models.normalization import RMSNorm
            import torch.nn as nn
            
            # Create new caption embedder with correct dimensions while preserving timestep_embedder
            new_caption_embedder_sequential=nn.Sequential(
                nn.Linear(
                    required_cap_feat_dim,current_cap_feat_dim,bias=True
                ),
                nn.GELU(),
                nn.Linear(
                    current_cap_feat_dim,current_cap_feat_dim,bias=True
                ),
                transformer.time_caption_embed.caption_embedder,
            )
            #new_caption_embedder_sequential = nn.Sequential(
            #    RMSNorm(required_cap_feat_dim, eps=float(transformer.time_caption_embed.caption_embedder[0].eps)),
            #    nn.Linear(required_cap_feat_dim, transformer.config.hidden_size, bias=True)
            #)
            
            # Replace only the caption_embedder part, keeping time_proj and timestep_embedder
            transformer.time_caption_embed.caption_embedder = new_caption_embedder_sequential
            
            # Update the config to reflect the new cap_feat_dim
            transformer.config.cap_feat_dim = required_cap_feat_dim
            
            logger.info(f"Successfully reinitialized caption_embedder with cap_feat_dim={required_cap_feat_dim}")
        else:
            logger.info(f"Transformer's cap_feat_dim ({current_cap_feat_dim}) matches text encoder dimension. No reinitialization needed.")
        import safetensors
        state_dict={}
        file_list=os.listdir(os.path.join(args.pretrained_model_name_or_path,"transformer"))
        for file_name in file_list:
            if file_name.endswith(".safetensors"):
                file_path=os.path.join(args.pretrained_model_name_or_path,"transformer",file_name)
                part_state_dict=safetensors.torch.load_file(file_path,device="cpu")
                state_dict.update(part_state_dict)
        transformer.load_state_dict(state_dict)
    elif args.stage_two and args.pretrained_model_name_or_path:
        transformer =Lumina2Transformer2DModel(os.path.join(args.pretrained_model_name_or_path,"transformer","config.json"))
        current_cap_feat_dim = transformer.config.cap_feat_dim
        required_cap_feat_dim = args.cross_attention_dim
        if current_cap_feat_dim != required_cap_feat_dim:
            logger.warning(
                f"Transformer's cap_feat_dim ({current_cap_feat_dim}) doesn't match text encoder dimension ({required_cap_feat_dim}). "
                f"Reinitializing caption_embedder..."
            )
            
            # Reinitialize only the caption embedder part, keeping timestep_embedder unchanged
            from diffusers.models.normalization import RMSNorm
            import torch.nn as nn
            
            # Create new caption embedder with correct dimensions while preserving timestep_embedder
            new_caption_embedder_sequential=nn.Sequential(
                nn.Linear(
                    required_cap_feat_dim,current_cap_feat_dim,bias=True
                ),
                nn.GELU(),
                nn.Linear(
                    current_cap_feat_dim,current_cap_feat_dim,bias=True
                ),
                transformer.time_caption_embed.caption_embedder,
            )
            #new_caption_embedder_sequential = nn.Sequential(
            #    RMSNorm(required_cap_feat_dim, eps=float(transformer.time_caption_embed.caption_embedder[0].eps)),
            #    nn.Linear(required_cap_feat_dim, transformer.config.hidden_size, bias=True)
            #)
            
            # Replace only the caption_embedder part, keeping time_proj and timestep_embedder
            transformer.time_caption_embed.caption_embedder = new_caption_embedder_sequential
            
            # Update the config to reflect the new cap_feat_dim
            transformer.config.cap_feat_dim = required_cap_feat_dim
            
            logger.info(f"Successfully reinitialized caption_embedder with cap_feat_dim={required_cap_feat_dim}")
        else:
            logger.info(f"Transformer's cap_feat_dim ({current_cap_feat_dim}) matches text encoder dimension. No reinitialization needed.")

    elif args.from_scratch and  args.pretrained_model_name_or_path :
       
        import json
        with open(os.path.join(args.pretrained_model_name_or_path,"transformer","config.json"),'r',encoding='utf-8') as f:
            config=json.load(f)
        config['cap_feat_dim']=args.cross_attention_dim
        if args.use_avg_norm_layer:
            config.update({
                'use_avg_norm_layer': True,
                'avg_norm_num_layers': num_text_layers,
                'avg_norm_hidden_size': text_encoder_dim,
                'avg_norm_eps': 1e-5,
                'avg_norm_trainable_weights': args.avg_norm_trainable_weights  # 传递新参数
            })
        
        transformer = Lumina2TransformerWithAvgNorm(**config)

    

        #transformer = Lumina2Transformer2DModel(**config)
    
    else:
        # Create from scratch with Lumina2 configuration
        transformer = Lumina2Transformer2DModel(
            sample_size=args.sample_size,
            patch_size=args.patch_size,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            hidden_size=2304,
            num_layers=args.num_layers,
            num_refiner_layers=2,
            num_attention_heads=args.num_attention_heads,
            num_kv_heads=8,
            multiple_of=256,
            ffn_dim_multiplier=None,
            norm_eps=1e-5,
            scaling_factor=1.0,
            axes_dim_rope=(16, args.sample_size, args.sample_size),
            axes_lens=(1024, args.resolution // 8, args.resolution // 8),
            cap_feat_dim=args.cross_attention_dim,
        )



    # 优先通过模型 API 一次性设置；若不可用则逐模块设置


    # freeze parameters of models to save more memory
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 确定参数精度
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # 关键：若使用 DeepSpeed 但未在其配置中启用对应精度，则回退到 float32，防止 dtype 冲突
    ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if ds_plugin is not None:
        ds_cfg = getattr(ds_plugin, "deepspeed_config", {}) or {}
        ds_bf16 = bool(ds_cfg.get("bf16", {}).get("enabled", False))
        ds_fp16 = bool(ds_cfg.get("fp16", {}).get("enabled", False))
        if accelerator.mixed_precision == "bf16" and not ds_bf16:
            logger.warning("DeepSpeed 未启用 bf16，回退为 float32 权重以避免 dtype 冲突。请在 deepspeed_config 启用 bf16.enabled 或改为 no 混合精度。")
            weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16" and not ds_fp16:
            logger.warning("DeepSpeed 未启用 fp16，回退为 float32 权重以避免 dtype 冲突。请在 deepspeed_config 启用 fp16.enabled 或改为 no 混合精度。")
            weight_dtype = torch.float32

    # 将模型搬到设备并使用最终的 weight_dtype

    #transformer = transformer.to(dtype=weight_dtype)
    
    # 2. VAE - 使用训练精度并移到设备
    vae = vae.to(dtype=weight_dtype, device=accelerator.device)
    
    # 3. 文本编码器 - 使用稳定精度，一次性转换
    text_encoder = text_encoder.to(dtype=weight_dtype, device=accelerator.device)
    text_encoder.eval()
    vae.eval()


    # 保持原生 PyTorch 注意力实现

 

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Prefer loading from snapshot if provided and present


    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            # Check if train_data_dir is a JSON file or a directory
            if args.train_data_dir.endswith('.json'):
                # If it's a JSON file, load it directly
                data_files["train"] = args.train_data_dir
                dataset = load_dataset(
                    "json",
                    data_files=data_files,
                    cache_dir=args.cache_dir,
                )
            else:
                dataset=load_dataset(args.train_data_dir)
                print(f"{args.local_rank}:sucess load data",flush=True)
        # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    preprocess_train = PreprocessTrain(
        image_column=image_column,
        caption_column=caption_column,
        tokenizer=tokenizer,
        caption_max_length=args.caption_max_length,
        train_transforms=transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper())),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                #transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
        prompt_drop_probability=args.prompt_drop_probability,
    )

    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    train_dataset = dataset["train"].with_transform(preprocess_train)

    print(f"{args.local_rank}:sucess preprocess_data",flush=True)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        timeout=60,
    )
    print(f"{args.local_rank}:sucess build dataloader",flush=True)
    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )
    print(f"{args.local_rank}:sucess get lr scheduler",flush=True)
    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    print(f"{args.local_rank}:sucess prepare accelerator",flush=True)
  
    if args.use_ema:
        ema_transformer = EMAModel(transformer.parameters(), model_cls=Lumina2Transformer2DModel, model_config=transformer.config)
        if args.offload_ema:
            ema_transformer.pin_memory()
        else:
            ema_transformer.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # 移除validation_prompts避免序列化问题
        if "validation_prompts" in tracker_config:
            tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    # 新增：训练速度监控变量
    start_time = time.time()
    last_log_time = start_time
    last_log_step = 0
    resume_step = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch

    else:
        initial_global_step = 0



    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        if epoch == first_epoch and resume_step > 0:
            # 跳过已处理的batches
            print(f"Skipping first {resume_step} batches in epoch {epoch}")
            skipped_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step*args.gradient_accumulation_steps)
            active_dataloader = skipped_dataloader
        else:
            active_dataloader = train_dataloader
        
        for step, batch in enumerate(active_dataloader):
            # 关键修复：检查batch是否有效，并在所有进程间同步
            # 使用一个标志来标记当前进程的batch是否有效
            batch_valid = (batch is not None)
            
            # 将布尔值转换为tensor以便在进程间传递
            batch_valid_tensor = torch.tensor([1 if batch_valid else 0], device=accelerator.device)
            
            # 在所有进程间广播并检查：如果任何进程的batch无效，所有进程都跳过
            # 使用all_reduce来检查是否所有进程都有有效batch
            torch.distributed.all_reduce(batch_valid_tensor, op=torch.distributed.ReduceOp.MIN)
            
            # 如果任何进程的batch无效，所有进程都跳过这个batch
            if batch_valid_tensor.item() == 0:
                logger.warning(f"至少一个进程的batch无效，所有进程跳过此batch (step {step})")
                continue
            
            with accelerator.accumulate(transformer):
                try:
                    # ...existing code...
                    batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                    batch["input_ids"] = batch["input_ids"].to(accelerator.device, dtype=torch.long, non_blocking=True)
                    batch["attention_mask"] = batch["attention_mask"].to(accelerator.device, dtype=torch.long, non_blocking=True)
                    # ...existing code...
                    encoder_attention_mask = batch["attention_mask"]
                    valid_token_counts = batch["attention_mask"].sum(dim=1)

                    # 修复：同步检查是否所有进程都有有效的token
                    has_valid_tokens = (valid_token_counts > 0).any().int()
                    has_valid_tokens_tensor = torch.tensor([has_valid_tokens], device=accelerator.device)
                    torch.distributed.all_reduce(has_valid_tokens_tensor, op=torch.distributed.ReduceOp.MIN)
                    
                    if has_valid_tokens_tensor.item() == 0:
                        logger.warning(f"至少一个进程没有有效tokens，所有进程跳过此batch (global_step {global_step})")
                        continue
                    
                    # ...existing code...
                    if encoder_attention_mask.dim() > 2:
                        encoder_attention_mask = encoder_attention_mask.view(encoder_attention_mask.shape[0], -1)
                    encoder_attention_mask = (encoder_attention_mask > 0)
                    
                    # ...existing code for text encoding and training...
                    with torch.no_grad():
                        if hasattr(text_encoder, 'encoder'):
                            encoder_outputs = text_encoder.encoder(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                output_hidden_states=True,
                                return_dict=True
                            )
                            all_hidden_states = encoder_outputs.hidden_states
                        else:
                            encoder_outputs = text_encoder(
                                batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                output_hidden_states=True,
                                return_dict=True
                            )
                            all_hidden_states = encoder_outputs.hidden_states
                        
                        if hasattr(transformer, 'use_avg_norm_layer') and transformer.use_avg_norm_layer:
                            if isinstance(all_hidden_states, tuple):
                                all_hidden_states = list(all_hidden_states)
                            all_hidden_states = [state.to(dtype=weight_dtype) for state in all_hidden_states]
                            encoder_hidden_states = all_hidden_states
                        else:
                            if args.feature_layer is None:
                                if isinstance(encoder_outputs, tuple):
                                    encoder_hidden_states = encoder_outputs[0]
                                else:
                                    encoder_hidden_states = encoder_outputs.last_hidden_state
                            else:
                                encoder_hidden_states=encoder_outputs.hidden_states[args.feature_layer]
                            encoder_hidden_states = encoder_hidden_states.to(dtype=weight_dtype)

                    # ...existing code...
                    ds_active = getattr(accelerator.state, "deepspeed_plugin", None) is not None
                    use_autocast = (accelerator.mixed_precision in ["fp16", "bf16"]) and (accelerator.device.type != "cpu") and (not ds_active)
                    autocast_ctx = torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype) if use_autocast else nullcontext()

                    with autocast_ctx:
                        model_input = vae.encode(batch["pixel_values"]).latent_dist.sample()
                        vae_config_scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
                        vae_config_shift_factor = getattr(vae.config, "shift_factor", 0.0)
                        model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor

                        noise = torch.randn_like(model_input)
                        if args.noise_offset:
                            noise += args.noise_offset * torch.randn(
                                (model_input.shape[0], model_input.shape[1], 1, 1), device=model_input.device
                            )
                        if args.input_perturbation:
                            new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                        
                        bsz = model_input.shape[0]
                        t0=0
                        t1=1
                        t = torch.normal(mean=0.0, std=1.0, size=(bsz,))
                        u = 1 / (1 + torch.exp(-t)) * (t1 - t0) + t0
                        u=u.to(model_input.device)
                        timesteps = (u * noise_scheduler.config.num_train_timesteps).long()

                        if args.input_perturbation:
                            noisy_latents = (1 - u.view(-1, 1, 1, 1)) * new_noise + u.view(-1, 1, 1, 1) * model_input
                        else:
                            noisy_latents =  (1-u.view(-1, 1, 1, 1)) * noise + u.view(-1, 1, 1, 1) * model_input
                        
                        noisy_latents = noisy_latents.to(dtype=weight_dtype)
                        target = model_input - noise

                        if args.dream_training:
                            noisy_latents, target = compute_dream_and_update_latents(
                                transformer,
                                noise_scheduler,
                                timesteps,
                                noise,
                                noisy_latents,
                                target,
                                encoder_hidden_states,
                                args.dream_detail_preservation,
                            )

                        model_pred = transformer(
                            noisy_latents,
                            u,
                            encoder_hidden_states,
                            encoder_attention_mask,
                            return_dict=False,
                        )[0]

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        grad_norm = None
                        grad_norm = accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        if args.use_ema:
                            try:
                                if args.offload_ema:
                                    ema_transformer.to(device=accelerator.device, non_blocking=True)
                                ema_transformer.step(transformer.parameters())
                                if args.offload_ema:
                                    ema_transformer.to(device="cpu", non_blocking=True)
                            except Exception as e:
                                logger.warning(f"EMA update failed at step {global_step}: {e}")

                        global_step += 1
                        
                        # ...existing code for metrics...
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        recent_elapsed = current_time - last_log_time
                        recent_steps = global_step - last_log_step
                        
                        steps_per_second = recent_steps / recent_elapsed if recent_elapsed > 0 else 0
                        samples_per_second = steps_per_second * total_batch_size
                        
                        log_dict = {"train_loss": train_loss}
                        
                        if grad_norm is not None:
                            log_dict["train/grad_norm"] = grad_norm
                        
                        log_dict.update({
                            "train/steps_per_second": steps_per_second,
                            "train/samples_per_second": samples_per_second,
                            "train/elapsed_time_hours": elapsed_time / 3600,
                        })
                        
                        accelerator.log(log_dict, step=global_step)
                        train_loss = 0.0

                        if accelerator.is_main_process and global_step % 1 == 0:
                            if torch.cuda.is_available():
                                accelerator.log({
                                    "memory/allocated_gb": torch.cuda.memory_allocated() / 1e9,
                                    "memory/reserved_gb": torch.cuda.memory_reserved() / 1e9,
                                    "memory/max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                                }, step=global_step)
                        
                            accelerator.log({
                                "train/epoch": epoch,
                                "train/global_step": global_step,
                                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            }, step=global_step)
                        
                        if global_step % 10 == 0:
                            last_log_time = current_time
                            last_log_step = global_step

                        # 关键修复：在保存checkpoint前同步所有进程
                        if global_step % args.checkpointing_steps == 0:
                            # 同步所有进程，确保都到达这个点
                            accelerator.wait_for_everyone()
                            
                            torch.cuda.empty_cache()
                            if accelerator.is_main_process:
                                if args.checkpoints_total_limit is not None:
                                    checkpoints = os.listdir(args.output_dir)
                                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                    if len(checkpoints) >= args.checkpoints_total_limit:
                                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                        removing_checkpoints = checkpoints[0:num_to_remove]

                                        logger.info(
                                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                        )
                                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                        for removing_checkpoint in removing_checkpoints:
                                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                            shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                            
                            # 再次同步，确保所有进程都完成保存
                            accelerator.wait_for_everyone()
                            torch.cuda.empty_cache()
                            
                except Exception as e:
                    logger.error(f"Training step failed at global_step {global_step}: {e}")
                    # 关键修复：同步错误状态到所有进程
                    error_occurred = torch.tensor([1], device=accelerator.device)
                    torch.distributed.all_reduce(error_occurred, op=torch.distributed.ReduceOp.MAX)
                    
                    # 清理GPU内存
                    torch.cuda.empty_cache()
                    
                    # 如果错误严重，考虑重新抛出异常让所有进程都知道
                    if "CUDA out of memory" in str(e):
                        logger.critical(f"OOM错误，需要所有进程停止: {e}")
                        raise e
                    continue

            if global_step >= args.max_train_steps:
                break
        
        # 关键修复：在epoch结束时同步所有进程
        accelerator.wait_for_everyone()
        
        # Create the pipeline using the trained modules and save it.
        if accelerator.is_main_process:
            try:
                # 只保存transformer，不保存整个pipeline
                transformer.save_pretrained(os.path.join(args.output_dir,"epoch"+str(epoch), "transformer"))
                logger.info(f"Transformer已成功保存到: {os.path.join(args.output_dir, 'transformer')}")
                
                # 保存模型配置信息
                config_dict = {
                    "model_type": "lumina2_transformer",
                    "text_encoder_name": args.text_encoder_name,
                    "vae_name": args.vae_name,
                    "resolution": args.resolution,
                    "cross_attention_dim": args.cross_attention_dim,
                    "training_steps": global_step,
                    "learning_rate": args.learning_rate,
                }
                
                import json
                with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
                    json.dump(config_dict, f, indent=2)
                logger.info("训练配置已保存")
                
            except Exception as save_error:
                logger.error(f"保存Transformer失败: {save_error}")
                # 尝试备用保存方法
                try:
                    torch.save(transformer.state_dict(), os.path.join(args.output_dir, "transformer_state_dict.pth"))
                    logger.info("使用备用方法保存Transformer state_dict成功")
                except Exception as e:
                    logger.error(f"备用保存方法也失败: {e}")
        
        # 再次同步，确保主进程保存完成后其他进程才继续
        accelerator.wait_for_everyone()

        if args.push_to_hub:
            save_model_card(args, repo_id, images, repo_folder=args.output_dir)
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()