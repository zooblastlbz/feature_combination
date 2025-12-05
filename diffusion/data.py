import random
import functools
from pathlib import Path
import os

from datasets import load_dataset
from datasets.fingerprint import Hasher
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


def _is_main_process():
    """判断是否为主进程"""
    return _get_process_rank() == 0


def tokenize_captions(examples, tokenizer, text_key, instruction, max_length, 
                      apply_chat_template, add_generation_prompt):
    """
    预先 tokenize 所有 captions（用于 datasets.map）
    不考虑 drop rate，只 tokenize 完整的 caption
    """
    captions = []
    
    for text in examples[text_key]:
        caption = instruction + str(text)
        captions.append(caption)
    
    if apply_chat_template:
        all_input_ids = []
        all_attention_masks = []
        
        for caption in captions:
            tokenized = tokenizer.apply_chat_template(
                [{"role": "user", "content": caption}],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
                add_generation_prompt=add_generation_prompt,
                return_dict=True,
            )
            all_input_ids.append(tokenized["input_ids"].squeeze(0).tolist())
            all_attention_masks.append(tokenized["attention_mask"].squeeze(0).tolist())
        
        examples["input_ids"] = all_input_ids
        examples["attention_mask"] = all_attention_masks
    else:
        tokenized = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        examples["input_ids"] = tokenized.input_ids.tolist()
        examples["attention_mask"] = tokenized.attention_mask.tolist()
    
    return examples


class PrecomputedTokenDataset(Dataset):
    """
    预计算 tokenized 结果的数据集
    
    使用 datasets.map() 预先计算所有 tokenized 结果，
    __getitem__ 中只需要加载图像和获取预计算的 tokens
    """
    def __init__(self, dataset, hparams, image_key, text_key):
        """
        Args:
            dataset: 已经预计算 tokenized 结果的 HuggingFace Dataset（已设置 torch format）
            hparams: 训练超参数
            image_key: 图像路径的键名
            text_key: 文本的键名
        """
        self.dataset = dataset
        self.hparams = hparams
        self.image_key = image_key
        self.text_key = text_key
        self.image_root = Path(hparams.data.image_root) if getattr(hparams.data, 'image_root', None) else None
        
        self.max_load_attempts = getattr(hparams.data, 'max_load_attempts', 3)
        
        self.transform = transforms.Compose([
            transforms.Resize(hparams.data.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            (transforms.CenterCrop(hparams.data.resolution) if hparams.data.center_crop 
             else transforms.RandomCrop(hparams.data.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self._placeholder_image = None
    
    def __len__(self):
        return len(self.dataset)
    
    def _get_placeholder_image(self):
        """获取占位符图像（懒加载）"""
        if self._placeholder_image is None:
            self._placeholder_image = Image.new(
                'RGB', 
                (self.hparams.data.resolution, self.hparams.data.resolution), 
                (128, 128, 128)
            )
        return self._placeholder_image.copy()
    
    def _load_image(self, image_path):
        """简化的图像加载方法"""
        try:
            image = Image.open(image_path)
            image.load()
            return image.convert('RGB')
        except Exception:
            return None
    
    def __getitem__(self, idx):
        original_idx = idx
        image = None
        item = None
        
        for attempt in range(self.max_load_attempts):
            item = self.dataset[idx]
            
            image_path = item[self.image_key]
            if self.image_root:
                image_path = str(self.image_root / image_path)
            
            image = self._load_image(image_path)
            
            if image is not None:
                break
            elif attempt < self.max_load_attempts - 1:
                idx = random.randint(0, len(self.dataset) - 1)
        
        if image is None:
            image = self._get_placeholder_image()
            item = self.dataset[original_idx]
        
        pixel_values = self.transform(image)
        
        # 优化点2: 使用 set_format("torch") 后，直接获取 tensor，无需转换
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def create_collate_fn(random_dropping_rate, empty_input_ids, empty_attention_mask):
    """
    创建带有动态 caption dropping 的 collate 函数
    
    Args:
        random_dropping_rate: caption drop 的概率（用于 CFG）
        empty_input_ids: 预计算的空 caption input_ids
        empty_attention_mask: 预计算的空 caption attention_mask
    """
    def collate_fn(examples):
        """数据集 collate 函数，支持动态 caption dropping"""
        batch_size = len(examples)
        
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        
        # 优化点3: 向量化 drop 判断，避免循环中的 random.random()
        if random_dropping_rate > 0:
            drop_mask = torch.rand(batch_size) < random_dropping_rate
        else:
            drop_mask = None
        
        input_ids_list = []
        attention_mask_list = []
        
        for i, ex in enumerate(examples):
            if drop_mask is not None and drop_mask[i]:
                # 优化点1: 移除 clone()，torch.stack 会自动复制数据
                input_ids_list.append(empty_input_ids)
                attention_mask_list.append(empty_attention_mask)
            else:
                input_ids_list.append(ex["input_ids"])
                attention_mask_list.append(ex["attention_mask"])
        
        input_ids = torch.stack(input_ids_list)
        input_ids = input_ids.to(memory_format=torch.contiguous_format)
        
        attention_mask = torch.stack(attention_mask_list)
        attention_mask = attention_mask.to(memory_format=torch.contiguous_format)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    return collate_fn


def _compute_empty_caption_tokens(tokenizer, instruction, max_length, apply_chat_template, add_generation_prompt):
    """
    预计算空 caption 的 tokens（用于 CFG dropping）
    """
    empty_caption = instruction if instruction else ""
    
    if apply_chat_template:
        if empty_caption:
            tokenized = tokenizer.apply_chat_template(
                [{"role": "user", "content": empty_caption}],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
                add_generation_prompt=add_generation_prompt,
                return_dict=True,
            )
            empty_input_ids = tokenized["input_ids"].squeeze(0)
            empty_attention_mask = tokenized["attention_mask"].squeeze(0)
        else:
            tokenized = tokenizer(
                "",
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            empty_input_ids = tokenized.input_ids.squeeze(0)
            empty_attention_mask = tokenized.attention_mask.squeeze(0)
            if empty_attention_mask.sum() == 0:
                empty_attention_mask[0] = 1
    else:
        tokenized = tokenizer(
            empty_caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        empty_input_ids = tokenized.input_ids.squeeze(0)
        empty_attention_mask = tokenized.attention_mask.squeeze(0)
        if empty_attention_mask.sum() == 0:
            empty_attention_mask[0] = 1
    
    return empty_input_ids, empty_attention_mask


def get_dataloader(hparams):
    """
    创建本地 JSON 数据集的 DataLoader
    
    使用 datasets.map() 预先计算所有 tokenized 结果
    训练时在 collate_fn 中动态进行 caption dropping
    """
    tokenizer = AutoTokenizer.from_pretrained(hparams.data.tokenizer)
    
    image_key = getattr(hparams.data, 'image_key', 'image')
    text_key = getattr(hparams.data, 'text_key', 'text')
    
    instruction = getattr(hparams.data, 'instruction', '')
    apply_chat_template = getattr(hparams.data, 'apply_chat_template', False)
    add_generation_prompt = getattr(hparams.data, 'add_generation_prompt', False)
    random_dropping_rate = getattr(hparams.data, 'random_dropping_rate', 0.0)
    
    if apply_chat_template and instruction:
        instruction_length = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction.rstrip()}],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=hparams.data.max_prompt_length,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
        )["input_ids"].shape[1] - 1
    elif instruction:
        instruction_length = tokenizer(
            instruction.rstrip(),
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=hparams.data.max_prompt_length,
        ).input_ids.shape[1] - 1
    else:
        instruction_length = 0
    
    max_length = hparams.data.max_prompt_length + instruction_length
    
    if _is_main_process():
        print(f"📚 正在加载数据集 {hparams.data.data_path}...")
    
    dataset = load_dataset('json', data_files=hparams.data.data_path, split='train')
    
    if _is_main_process():
        print(f"✅ 成功加载 {len(dataset)} 个样本")
        print(f"📌 使用键名: image_key='{image_key}', text_key='{text_key}'")
    
    if image_key not in dataset.column_names:
        raise KeyError(f"❌ JSON 中找不到图像键 '{image_key}'")
    if text_key not in dataset.column_names:
        raise KeyError(f"❌ JSON 中找不到文本键 '{text_key}'")
    
    if _is_main_process():
        print(f"🔄 正在预计算 tokenized 结果...")
        print(f"  - max_length: {max_length}")
        print(f"  - apply_chat_template: {apply_chat_template}")
        print(f"  - random_dropping_rate: {random_dropping_rate} (将在训练时动态应用)")
    
    tokenize_fn = functools.partial(
        tokenize_captions,
        tokenizer=tokenizer,
        text_key=text_key,
        instruction=instruction,
        max_length=max_length,
        apply_chat_template=apply_chat_template,
        add_generation_prompt=add_generation_prompt,
    )
    
    cache_fingerprint = Hasher.hash({
        "tokenizer": hparams.data.tokenizer,
        "max_length": max_length,
        "instruction": instruction,
        "apply_chat_template": apply_chat_template,
        "add_generation_prompt": add_generation_prompt,
    })
    
    dataset_with_tokens = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        num_proc=min(8, os.cpu_count() or 1),
        new_fingerprint=cache_fingerprint,
        desc="Tokenizing captions",
    )
    
    if _is_main_process():
        print(f"✅ Tokenization 完成")
    
    # 优化点2: 设置 torch format，避免 __getitem__ 中的 torch.tensor() 转换
    dataset_with_tokens.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
        output_all_columns=True,  # 保留其他列（如 image_key）
    )
    
    if _is_main_process():
        print(f"✅ 已设置 torch format，避免运行时 tensor 转换")
    
    empty_input_ids, empty_attention_mask = _compute_empty_caption_tokens(
        tokenizer=tokenizer,
        instruction=instruction,
        max_length=max_length,
        apply_chat_template=apply_chat_template,
        add_generation_prompt=add_generation_prompt,
    )
    
    if _is_main_process():
        print(f"✅ 空 caption tokens 预计算完成 (用于 CFG dropping)")
    
    train_dataset = PrecomputedTokenDataset(
        dataset=dataset_with_tokens,
        hparams=hparams,
        image_key=image_key,
        text_key=text_key,
    )
    
    collate_fn = create_collate_fn(
        random_dropping_rate=random_dropping_rate,
        empty_input_ids=empty_input_ids,
        empty_attention_mask=empty_attention_mask,
    )
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(int(hparams.trainer.seed) + _get_process_rank())

    num_workers = hparams.data.dataloader_num_workers
    persistent_workers = getattr(hparams.data, 'persistent_workers', True) and num_workers > 0
    prefetch_factor = getattr(hparams.data, 'prefetch_factor', 4) if num_workers > 0 else None
    pin_memory = getattr(hparams.data, 'pin_memory', True)

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams.data.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        generator=g,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    
    if _is_main_process():
        print(f"✅ DataLoader 创建完成，共 {len(dataloader)} 个 batch")
        print(f"  - num_workers: {num_workers}")
        print(f"  - persistent_workers: {persistent_workers}")
        print(f"  - prefetch_factor: {prefetch_factor}")
        print(f"  - pin_memory: {pin_memory}")
    
    return dataloader
