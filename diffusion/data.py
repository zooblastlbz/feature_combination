import random
import json
from pathlib import Path
import os

from datasets import load_dataset
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


def _get_process_rank():
    """获取当前进程的 rank"""
    try:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


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
        
        # 错误处理配置
        self.max_load_attempts = getattr(hparams.data, 'max_load_attempts', 5)
        self.log_errors = getattr(hparams.data, 'log_image_errors', True)
        
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
        
        # 错误日志
        if self.log_errors:
            log_dir = Path(hparams.trainer.checkpoint_dir) / "data_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.error_log_path = log_dir / f"corrupted_images_rank_{_get_process_rank()}.txt"
            print(f"📝 错误日志: {self.error_log_path}")
        else:
            self.error_log_path = None
        
        self.error_count = 0
        
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
    
    def _load_image_robust(self, image_path):
        """鲁棒的图像加载方法"""
        try:
            if not Path(image_path).exists():
                return None, f"文件不存在: {image_path}"
            
            image = Image.open(image_path)
            image.load()
            image = image.convert('RGB')
            return image, None
            
        except Exception as e:
            return None, str(e)
    
    def _log_error(self, idx, image_path, error_msg):
        """记录错误到日志文件"""
        if not self.log_errors or self.error_log_path is None:
            return
        
        try:
            with open(self.error_log_path, 'a') as f:
                f.write(f"{idx}\t{image_path}\t{error_msg}\n")
        except Exception:
            pass
    
    def _create_placeholder_image(self):
        """创建灰色占位符图像"""
        return Image.new(
            'RGB', 
            (self.hparams.data.resolution, self.hparams.data.resolution), 
            (128, 128, 128)
        )
    
    def __getitem__(self, idx):
        original_idx = idx
        image = None
        item = None
        
        for attempt in range(self.max_load_attempts):
            try:
                item = self.dataset[idx]
                
                image_path = item[self.image_key]
                if self.image_root:
                    image_path = self.image_root / image_path
                else:
                    image_path = Path(image_path)
                
                image, error = self._load_image_robust(image_path)
                
                if image is not None:
                    break
                else:
                    self.error_count += 1
                    
                    if attempt == 0:
                        self._log_error(original_idx, str(image_path), error)
                    
                    if attempt < self.max_load_attempts - 1:
                        idx = random.randint(0, len(self.dataset) - 1)
                        if attempt == 0:
                            print(f"⚠️ [{self.error_count}] 无法加载图像 {image_path}: {error}")
                    else:
                        image = self._create_placeholder_image()
                        item = self.dataset[original_idx]
                        break
                        
            except Exception as e:
                self.error_count += 1
                print(f"❌ 处理样本 {idx} 时发生未预期错误: {e}")
                
                if attempt < self.max_load_attempts - 1:
                    idx = random.randint(0, len(self.dataset) - 1)
                else:
                    image = self._create_placeholder_image()
                    item = self.dataset[original_idx]
                    break
        
        if image is None:
            image = self._create_placeholder_image()
            item = self.dataset[original_idx]
        
        pixel_values = self.transform(image)
        
        caption = str(item[self.text_key])
        
        # 随机丢弃 caption（用于 CFG 训练）
        if random.random() < self.hparams.data.random_dropping_rate:
            caption = ""

            
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


def collate_fn(examples):
    """数据集 collate 函数"""
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


def get_dataloader(hparams):
    """
    创建本地 JSON 数据集的 DataLoader
    
    在 YAML 配置文件中设置:
    data:
      data_path: "path/to/your/data.json"
      image_root: "path/to/images"  # 可选
      image_key: "image"
      text_key: "text"
    """
    tokenizer = AutoTokenizer.from_pretrained(hparams.data.tokenizer)
    
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
        instruction = getattr(hparams.data, 'instruction', '')
        if instruction:
            hparams.data.instruction_length = tokenizer(
                instruction.rstrip(),
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=hparams.data.max_prompt_length,
            ).input_ids.shape[1] - 1
        else:
            hparams.data.instruction_length = 0
    
    # 创建数据集
    image_root = getattr(hparams.data, 'image_root', None)
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
    
    g = torch.Generator()
    g.manual_seed(int(hparams.trainer.seed) + _get_process_rank())

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=hparams.data.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=hparams.data.dataloader_num_workers,
        generator=g,
        worker_init_fn=seed_worker,
        pin_memory=True,
        drop_last=True,
    )
    
    print(f"✅ DataLoader 创建完成，共 {len(dataloader)} 个 batch")
    return dataloader