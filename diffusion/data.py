from functools import partial
import random
import json
from pathlib import Path

from datasets import load_dataset
from diffusers.utils import is_torch_xla_available
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer, GemmaTokenizer
import webdataset as wds


class LocalImageTextDataset(Dataset):
    """
    本地 JSON 格式数据集加载器（使用 HuggingFace datasets）
    
    支持的 JSON 格式:
    [
        {"image": "path/to/image1.jpg", "text": "caption for image 1"},
        {"image": "path/to/image2.jpg", "text": "caption for image 2"},
        ...
    ]
    
    可以通过 hparams.data.image_key 和 hparams.data.text_key 自定义键名
    
    使用 datasets 库的优势：
    - 自动缓存和内存映射
    - 支持流式加载大型数据集
    - 更好的性能和内存管理
    """
    def __init__(self, json_path, hparams, tokenizer, image_root=None):
        """
        Args:
            json_path: JSON 文件路径
            hparams: 训练超参数
            tokenizer: 文本分词器
            image_root: 图像根目录（如果 JSON 中的路径是相对路径）
        """
        self.hparams = hparams
        self.tokenizer = tokenizer
        self.image_root = Path(image_root) if image_root else None
        
        # 从配置中获取键名，提供默认值
        self.image_key = getattr(hparams.data, 'image_key', 'image')
        self.text_key = getattr(hparams.data, 'text_key', 'text')
        
        # 使用 datasets 加载 JSON 数据
        print(f"📚 正在使用 datasets 库加载 {json_path}...")
        self.dataset = load_dataset('json', data_files=json_path, split='train')
        
        print(f"✅ 成功加载 {len(self.dataset)} 个样本从 {json_path}")
        print(f"📌 使用键名: image_key='{self.image_key}', text_key='{self.text_key}'")
        print(f"📊 数据集特征: {self.dataset.column_names}")
        
        # 验证数据集是否包含指定的键
        if self.image_key not in self.dataset.column_names:
            raise KeyError(
                f"❌ JSON 中找不到图像键 '{self.image_key}'，"
                f"可用的键: {self.dataset.column_names}"
            )
        if self.text_key not in self.dataset.column_names:
            raise KeyError(
                f"❌ JSON 中找不到文本键 '{self.text_key}'，"
                f"可用的键: {self.dataset.column_names}"
            )
        
        # 图像预处理变换
        self.transform = transforms.Compose([
            transforms.Resize(hparams.data.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            (transforms.CenterCrop(hparams.data.resolution) if hparams.data.center_crop 
             else transforms.RandomCrop(hparams.data.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 使用 datasets 获取样本
        item = self.dataset[idx]
        
        # 使用配置的键名加载图像
        image_path = item[self.image_key]
        if self.image_root:
            image_path = self.image_root / image_path
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 无法加载图像 {image_path}: {e}")
            # 返回一个纯黑图像作为占位符
            image = Image.new('RGB', (self.hparams.data.resolution, self.hparams.data.resolution), (0, 0, 0))
        
        # 应用图像变换
        pixel_values = self.transform(image)
        
        # 使用配置的键名获取文本
        caption = str(item[self.text_key])  # 确保是字符串类型
        
        # 随机丢弃 caption（用于 CFG 训练）
        if random.random() < self.hparams.data.random_dropping_rate:
            caption = ""
        else:
            # 添加 instruction 前缀
            instruction = getattr(self.hparams.data, 'instruction', '')
            caption = instruction + caption
        
        # Tokenize 文本
        if hasattr(self.hparams.data, 'apply_chat_template') and self.hparams.data.apply_chat_template and caption != "":
            tokenized = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": caption}],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.hparams.data.max_prompt_length + self.hparams.data.instruction_length,
                add_generation_prompt=self.hparams.data.add_generation_prompt,
                return_dict=True,
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
        else:
            tokenized = self.tokenizer(
                caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.hparams.data.max_prompt_length + self.hparams.data.instruction_length,
            )
            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def get_local_json_dataloader(hparams, *args, **kwargs):
    """
    创建本地 JSON 数据集的 DataLoader
    
    在 YAML 配置文件中添加:
    data:
      data_path: "path/to/your/data.json"  # JSON 文件路径
      image_root: "path/to/images"          # 可选，图像根目录
      use_local_json: true                  # 启用本地 JSON 加载
    """
    tokenizer = GemmaTokenizer.from_pretrained(hparams.data.tokenizer)
    
    # 计算 instruction 长度
    if hasattr(hparams.data, 'apply_chat_template') and hparams.data.apply_chat_template:
        hparams.data.instruction_length = tokenizer.apply_chat_template(
            [{"role": "user", "content": hparams.data.instruction.rstrip()}],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=hparams.data.max_prompt_length,
            add_generation_prompt=hparams.data.add_generation_prompt,
            return_dict=True,
        )["input_ids"].shape[1] - 1
    else:
        hparams.data.instruction_length = tokenizer(
            hparams.data.instruction.rstrip(),
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=hparams.data.max_prompt_length,
        ).input_ids.shape[1] - 1
    
    # 创建数据集
    image_root = hparams.data.image_root if hasattr(hparams.data, 'image_root') else None
    dataset = LocalImageTextDataset(
        json_path=hparams.data.data_path,
        hparams=hparams,
        tokenizer=tokenizer,
        image_root=image_root
    )
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=hparams.data.batch_size,
        shuffle=True,  # 本地数据集需要手动 shuffle
        collate_fn=llm_collate_fn,
        num_workers=hparams.data.dataloader_num_workers,
        generator=torch.manual_seed(hparams.trainer.seed),
        worker_init_fn=seed_worker,
        pin_memory=False if is_torch_xla_available() else True,
        drop_last=True,  # 丢弃最后不完整的 batch
    )


def llm_preprocess_fn(hparams, tokenizer, sample):
    image = sample[hparams.data.image_column]

    if hparams.data.original_caption_rate > 0 and sample.get(hparams.data.caption_column.original) is not None and random.random() < hparams.data.original_caption_rate:
        caption = sample[hparams.data.caption_column.original]
    else:
        caption = sample[hparams.data.caption_column.synthetic]

    transform = transforms.Compose(
        [
            transforms.Resize(hparams.data.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            (transforms.CenterCrop(hparams.data.resolution) if hparams.data.center_crop else transforms.RandomCrop(hparams.data.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    pixel_values = transform(image)

    if random.random() < hparams.data.random_dropping_rate: # Randomly drop the caption
        caption = ""
    else:
        caption = hparams.data.instruction + caption

    if hparams.data.apply_chat_template and caption != "":
        tokenized = tokenizer.apply_chat_template(
            [{ "role": "user", "content": caption }],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=hparams.data.max_prompt_length + hparams.data.instruction_length,
            add_generation_prompt=hparams.data.add_generation_prompt,
            return_dict=True,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
    else:
        tokenized = tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=hparams.data.max_prompt_length + hparams.data.instruction_length,
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def llm_collate_fn(examples):
    pixel_values = [example["pixel_values"] for example in examples]
    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    input_ids = [example["input_ids"] for example in examples]
    input_ids = torch.cat(input_ids).to(memory_format=torch.contiguous_format)
    attention_mask = [example["attention_mask"] for example in examples]
    attention_mask = torch.cat(attention_mask).to(memory_format=torch.contiguous_format)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def get_llm_dataloader(hparams, *args, **kwargs):
    tokenizer = GemmaTokenizer.from_pretrained(hparams.data.tokenizer)

    if hparams.data.apply_chat_template:
        hparams.data.instruction_length = tokenizer.apply_chat_template(
            [{ "role": "user", "content": hparams.data.instruction.rstrip() }],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=hparams.data.max_prompt_length,
            add_generation_prompt=hparams.data.add_generation_prompt,
            return_dict=True,
        )["input_ids"].shape[1] - 1
    else:
        hparams.data.instruction_length = tokenizer(
            hparams.data.instruction.rstrip(),
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=hparams.data.max_prompt_length,
        ).input_ids.shape[1] - 1

    dataset = (
        wds.WebDataset(wds.ResampledShards(hparams.data.data_path, deterministic=True))
            .shuffle(1000, rng=random.Random(hparams.trainer.seed))
            .decode("pil")
            .map(
                partial(
                    llm_preprocess_fn,
                    hparams,
                    tokenizer,
                ),
            )
    )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=hparams.data.batch_size,
        collate_fn=llm_collate_fn,
        num_workers=hparams.data.dataloader_num_workers,
        generator=torch.manual_seed(hparams.trainer.seed),
        worker_init_fn=seed_worker,
        pin_memory=False if is_torch_xla_available() else True,
        prefetch_factor=8,
    )


def clip_llm_preprocess_fn(hparams, clip_tokenizer, tokenizer, sample):
    image = sample[hparams.data.image_column]

    if hparams.data.original_caption_rate > 0 and sample.get(hparams.data.caption_column.original) is not None and random.random() < hparams.data.original_caption_rate:
        caption = sample[hparams.data.caption_column.original]
    else:
        caption = sample[hparams.data.caption_column.synthetic]

    transform = transforms.Compose(
        [
            transforms.Resize(hparams.data.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            (transforms.CenterCrop(hparams.data.resolution) if hparams.data.center_crop else transforms.RandomCrop(hparams.data.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    pixel_values = transform(image)

    if random.random() < hparams.data.random_dropping_rate: # Randomly drop the caption
        caption = ""
    else:
        caption = hparams.data.instruction + caption

    clip_input_ids = clip_tokenizer(
        caption,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,
    ).input_ids

    if hparams.data.apply_chat_template and caption != "":
        tokenized = tokenizer.apply_chat_template(
            [{ "role": "user", "content": caption }],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=hparams.data.max_prompt_length + hparams.data.instruction_length,
            add_generation_prompt=hparams.data.add_generation_prompt,
            return_dict=True,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
    else:
        tokenized = tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=hparams.data.max_prompt_length + hparams.data.instruction_length,
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

    return {
        "pixel_values": pixel_values,
        "clip_input_ids": clip_input_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def clip_llm_collate_fn(examples):
    pixel_values = [example["pixel_values"] for example in examples]
    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    clip_input_ids = [example["clip_input_ids"] for example in examples]
    clip_input_ids = torch.cat(clip_input_ids).to(memory_format=torch.contiguous_format)
    input_ids = [example["input_ids"] for example in examples]
    input_ids = torch.cat(input_ids).to(memory_format=torch.contiguous_format)
    attention_mask = [example["attention_mask"] for example in examples]
    attention_mask = torch.cat(attention_mask).to(memory_format=torch.contiguous_format)

    return {
        "pixel_values": pixel_values,
        "clip_input_ids": clip_input_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def get_clip_llm_dataloader(hparams, *args, **kwargs):
    clip_tokenizer = CLIPTokenizer.from_pretrained(**hparams.data.tokenizer.clip)
    tokenizer = GemmaTokenizer.from_pretrained(hparams.data.tokenizer.llm)

    hparams.data.instruction_length = tokenizer(
        hparams.data.instruction.rstrip(),
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=hparams.data.max_prompt_length,
    ).input_ids.shape[1] - 1

    dataset = (
        wds.WebDataset(wds.ResampledShards(hparams.data.data_path, deterministic=True))
            .shuffle(1000, rng=random.Random(hparams.trainer.seed))
            .decode("pil")
            .map(
                partial(
                    clip_llm_preprocess_fn,
                    hparams,
                    clip_tokenizer,
                    tokenizer,
                ),
            )
    )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=hparams.data.batch_size,
        collate_fn=clip_llm_collate_fn,
        num_workers=hparams.data.dataloader_num_workers,
        generator=torch.manual_seed(hparams.trainer.seed),
        worker_init_fn=seed_worker,
        pin_memory=False if is_torch_xla_available() else True,
        prefetch_factor=8,
    )


def get_dataloader(hparams, *args, **kwargs):
    # 检查是否使用本地 JSON 数据集
    if hasattr(hparams.data, 'use_local_json') and hparams.data.use_local_json:
        return get_local_json_dataloader(hparams, *args, **kwargs)
    
    if hparams.model.encoder_type == "clip-llm":
        return get_clip_llm_dataloader(hparams, *args, **kwargs)
    elif hparams.model.encoder_type == "llm":
        return get_llm_dataloader(hparams, *args, **kwargs)
    else:
        raise ValueError(f"Invalid encoder_type: {hparams.model.encoder_type}")