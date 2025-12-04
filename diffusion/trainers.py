from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from io import BytesIO
import math
import os
import random
import shutil
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import is_wandb_available
import numpy as np
from omegaconf import OmegaConf
import torch
from transformers import CLIPTextModelWithProjection

from .data import get_dataloader
from .models import build_model, get_llm, update_self_attention_mask

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def seed_everything(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer(ABC):
    """训练器基类"""
    
    def before_training(self):
        """训练前初始化"""
        if self.accelerator.is_main_process:
            if is_wandb_available():
                wandb.init(
                    project=self.hparams.trainer.project,
                    name=self.hparams.trainer.run,
                    config=OmegaConf.to_container(self.hparams, resolve=True)
                )
            print("***** Running training *****")
            print(f"  Total train batch size = {self.total_batch_size}")
            print(f"  Total optimization steps = {self.hparams.trainer.max_steps}")
            self.progress_bar = tqdm(total=self.hparams.trainer.max_steps, initial=self.global_step)

    @abstractmethod
    def backward(self, loss):
        pass

    @abstractmethod
    def optimizer_step(self):
        pass

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(dtype=dtype)
        step_indices = [(self.noise_scheduler.timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def dit_training_step(self, batch):
        """DiT 模型训练步骤"""
        pixel_values = batch["pixel_values"]

        # 1. LLM 推理（no_grad）
        if self.hparams.model.encoder_type == "llm":
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(self.accelerator.device)

            llm_attention_mask = update_self_attention_mask(
                attention_mask, 0, False, self.accelerator.device, 
                self.weight_dtype
            )

            position_ids = torch.arange(input_ids.shape[1], device=self.accelerator.device).unsqueeze(0)

            with torch.no_grad():
                text_hidden_states = self.llm(
                    input_ids.to(self.accelerator.device),
                    llm_attention_mask,
                    position_ids,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=False,
                )[1][self.hparams.model.text_hidden_states_index]
            text_hidden_states = text_hidden_states.to(dtype=self.weight_dtype)
        else:
            raise ValueError(f"Unknown encoder type: {self.hparams.model.encoder}")

        # 2. VAE 编码（no_grad，float32）
        model_input = pixel_values.to(self.accelerator.device)
        with torch.no_grad():
            model_input = self.vae.encode(model_input.float()).latent_dist.sample()
        model_input = (model_input - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        model_input = model_input.to(dtype=self.weight_dtype)

        # 3. DiT 前向
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.hparams.trainer.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.hparams.trainer.logit_mean,
            logit_std=self.hparams.trainer.logit_std,
            mode_scale=self.hparams.trainer.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices]

        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype).to(self.accelerator.device)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        noisy_model_input = noisy_model_input.to(dtype=self.weight_dtype)
        
        output = self.model(
            hidden_states=noisy_model_input,
            timestep=timesteps.to(self.accelerator.device),
            text_hidden_states=text_hidden_states,
            attention_mask=attention_mask,
        )
        model_pred = output

        if self.hparams.trainer.precondition_outputs:
            model_pred = model_pred * (-sigmas) + noisy_model_input

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.hparams.trainer.weighting_scheme, sigmas=sigmas)

        if self.hparams.trainer.precondition_outputs:
            target = model_input
        else:
            target = noise - model_input

        # Loss 计算（float32）
        loss = torch.mean((weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1)
        loss = loss.mean()

        return loss

    def fusedit_training_step(self, batch):
        """FuseDiT 模型训练步骤"""
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # 1. CLIP 推理（no_grad）
        if self.hparams.model.encoder_type == "clip-llm":
            clip_input_ids = batch["clip_input_ids"]
            with torch.no_grad():
                text_modulation_embeds = self.clip(clip_input_ids.to(self.accelerator.device)).text_embeds
            text_modulation_embeds = text_modulation_embeds.to(dtype=self.weight_dtype)
        else:
            text_modulation_embeds = None

        # 2. VAE 编码（no_grad，float32）
        model_input = pixel_values.to(self.accelerator.device)
        with torch.no_grad():
            model_input = self.vae.encode(model_input.float()).latent_dist.sample()
        model_input = (model_input - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        model_input = model_input.to(dtype=self.weight_dtype)

        # 3. FuseDiT 前向
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.hparams.trainer.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.hparams.trainer.logit_mean,
            logit_std=self.hparams.trainer.logit_std,
            mode_scale=self.hparams.trainer.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices]

        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype).to(self.accelerator.device)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        noisy_model_input = noisy_model_input.to(dtype=self.weight_dtype)
        
        output = self.model(
            hidden_states=noisy_model_input,
            timestep=timesteps.to(self.accelerator.device),
            input_ids=input_ids.to(self.accelerator.device),
            text_modulation_embeds=text_modulation_embeds,
            attention_mask=attention_mask.to(self.accelerator.device),
        )
        model_pred = output[0]

        if self.hparams.trainer.precondition_outputs:
            model_pred = model_pred * (-sigmas) + noisy_model_input

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.hparams.trainer.weighting_scheme, sigmas=sigmas)

        if self.hparams.trainer.precondition_outputs:
            target = model_input
        else:
            target = noise - model_input

        loss = torch.mean((weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1)
        loss = loss.mean()

        return loss

    def adafusedit_training_step(self, batch):
        """AdaFuseDiT 训练步骤"""
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].to(self.accelerator.device)

        # 1. LLM 推理（no_grad）
        llm_attention_mask = update_self_attention_mask(
            attention_mask, 0, False, self.accelerator.device, 
            self.weight_dtype
        )
        position_ids = torch.arange(input_ids.shape[1], device=self.accelerator.device).unsqueeze(0)

        with torch.no_grad():
            llm_output = self.llm(
                input_ids.to(self.accelerator.device),
                llm_attention_mask,
                position_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=False,
            )
        all_hidden_states = llm_output[1]
        
        text_hidden_states_num = getattr(self.hparams.model, 'text_hidden_states_num', 1)
        
        if text_hidden_states_num > 1:
            text_hidden_states = [
                all_hidden_states[-text_hidden_states_num + i].to(dtype=self.weight_dtype)
                for i in range(text_hidden_states_num)
            ]
        else:
            text_hidden_states_index = getattr(self.hparams.model, 'text_hidden_states_index', -1)
            text_hidden_states = all_hidden_states[text_hidden_states_index].to(dtype=self.weight_dtype)

        # 2. VAE 编码（no_grad，float32）
        model_input = pixel_values.to(self.accelerator.device)
        with torch.no_grad():
            model_input = self.vae.encode(model_input.float()).latent_dist.sample()
        model_input = (model_input - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        model_input = model_input.to(dtype=self.weight_dtype)

        # 3. AdaFuseDiT 前向
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.hparams.trainer.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.hparams.trainer.logit_mean,
            logit_std=self.hparams.trainer.logit_std,
            mode_scale=self.hparams.trainer.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices]

        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype).to(self.accelerator.device)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        noisy_model_input = noisy_model_input.to(dtype=self.weight_dtype)

        model_pred = self.model(
            hidden_states=noisy_model_input,
            timestep=timesteps.to(self.accelerator.device),
            text_hidden_states=text_hidden_states,
            attention_mask=attention_mask,
        )

        if self.hparams.trainer.precondition_outputs:
            model_pred = model_pred * (-sigmas) + noisy_model_input

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.hparams.trainer.weighting_scheme, sigmas=sigmas)

        if self.hparams.trainer.precondition_outputs:
            target = model_input
        else:
            target = noise - model_input

        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1
        )
        loss = loss.mean()

        return loss

    def log(self, data, progress_bar=True):
        """日志记录"""
        if self.accelerator.is_main_process:
            if is_wandb_available():
                wandb.log(data, step=self.global_step)
            if progress_bar:
                self.progress_bar.update(1)
                self.progress_bar.set_postfix(data)

    @torch.no_grad()
    def update_ema(self):
        """更新 EMA 模型"""
        if self.ema is None:
            return
        if self.hparams.ema.update_steps is not None and self.global_step % self.hparams.ema.update_steps == 0:
            self.accelerator.wait_for_everyone()
            
            ema_params = OrderedDict(self.ema.named_parameters())
            model_params = OrderedDict(self.accelerator.unwrap_model(self.model).named_parameters())
            assert set(ema_params.keys()) == set(model_params.keys())

            for name, param in [(k, v) for k, v in model_params.items() if v.requires_grad]:
                ema_params[name].mul_(self.hparams.ema.decay).add_(param.data, alpha=1 - self.hparams.ema.decay)

    @abstractmethod
    def save_checkpoint(self):
        pass

    def train(self):
        """主训练循环"""
        self.before_training()
        
        # 计算每个 epoch 的步数
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.hparams.trainer.gradient_accumulation_steps)
        num_train_epochs = math.ceil(self.hparams.trainer.max_steps / num_update_steps_per_epoch)
        
        starting_epoch = self.global_step // num_update_steps_per_epoch
        resume_step = self.global_step % num_update_steps_per_epoch

        for epoch in range(starting_epoch, num_train_epochs):
            if epoch == starting_epoch and resume_step > 0:
                active_dataloader = self.accelerator.skip_first_batches(
                    self.train_dataloader, 
                    resume_step * self.hparams.trainer.gradient_accumulation_steps
                )
                if self.accelerator.is_main_process:
                    print(f"⏭️ 跳过前 {resume_step * self.hparams.trainer.gradient_accumulation_steps} 个 batch")
            else:
                active_dataloader = self.train_dataloader
            
            for step, batch in enumerate(active_dataloader):
                with self.accelerator.accumulate(self.model):
                    loss = self.training_step(batch)
                    self.backward(loss)
                    self.optimizer_step()

                if self.accelerator.sync_gradients:
                    self.global_step += 1

                    if self.global_step % self.hparams.trainer.logging_steps == 0:
                        self.log({
                            "train/loss": loss.detach().item(), 
                            "train/lr": self.lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                        })
                    elif self.accelerator.is_main_process:
                        self.progress_bar.update(1)

                    self.update_ema()
                    self.save_checkpoint()

                    if self.global_step >= self.hparams.trainer.max_steps:
                        break
            
            if self.global_step >= self.hparams.trainer.max_steps:
                break

        self.after_training()

    def after_training(self):
        """训练结束后的清理工作"""
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print("Training completed!")
        self.accelerator.end_training()


class AccelerateTrainer(Trainer):
    """基于 HuggingFace Accelerate 的训练器"""
    
    def __init__(self, hparams):
        self.hparams = hparams
        self.global_step = 0

        # 1. 创建 Accelerator
        project_config = ProjectConfiguration(
            project_dir=hparams.trainer.checkpoint_dir,
            logging_dir=os.path.join(hparams.trainer.checkpoint_dir, "logs")
        )
        
        mixed_precision = hparams.trainer.mixed_precision
        if mixed_precision == "fp32":
            mixed_precision = "no"
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=hparams.trainer.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with="wandb" if is_wandb_available() else None,
            project_config=project_config,
        )

        if hparams.trainer.seed is not None:
            set_seed(hparams.trainer.seed)

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        self.total_batch_size = (
            hparams.data.batch_size * 
            self.accelerator.num_processes * 
            hparams.trainer.gradient_accumulation_steps
        )

        self.model = build_model(hparams)
        self.model.train()
        
        if hparams.trainer.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if hparams.ema.update_steps is not None:
            self.ema = deepcopy(self.model)
            self.ema.requires_grad_(False)
        else:
            self.ema = None

        self.vae = AutoencoderKL.from_pretrained(**hparams.vae)
        self.vae.requires_grad_(False)
        self.vae.to(self.accelerator.device, dtype=torch.float32)

        if hparams.model.name == "DiT":
            if hparams.model.encoder_type == "llm":
                self.llm = get_llm(hparams.model.base, self.model.config.base_config)
                self.llm.requires_grad_(False)
                self.llm.to(self.accelerator.device, dtype=self.weight_dtype)
            else:
                raise ValueError(f"Unknown encoder type: {hparams.model.encoder_type}")
            self.training_step = self.dit_training_step
            
        elif hparams.model.name == "AdaFuseDiT":
            if hparams.model.encoder_type == "llm":
                self.llm = get_llm(hparams.model.base, self.model.config.base_config)
                self.llm.requires_grad_(False)
                self.llm.to(self.accelerator.device, dtype=self.weight_dtype)
            else:
                raise ValueError(f"Unknown encoder type: {hparams.model.encoder_type}")
            self.training_step = self.adafusedit_training_step
            
        elif hparams.model.name == "FuseDiT":
            if hparams.model.encoder_type == "clip-llm":
                self.clip = CLIPTextModelWithProjection.from_pretrained(**hparams.clip_l)
                self.clip.requires_grad_(False)
                self.clip.to(self.accelerator.device, dtype=self.weight_dtype)
            self.training_step = self.fusedit_training_step
        else:
            raise ValueError(f"Unknown model name: {hparams.model.name}")

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(**hparams.noise_scheduler)

        self.optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(), 
            **hparams.optimizer
        )

        self.lr_scheduler = get_scheduler(
            **hparams.lr_scheduler, 
            optimizer=self.optimizer, 
            num_training_steps=hparams.trainer.max_steps * hparams.trainer.gradient_accumulation_steps
        )

        self.train_dataloader = get_dataloader(hparams)

        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        if self.ema is not None:
            self.ema.to(self.accelerator.device)

        self.load_checkpoint()

        if self.accelerator.is_main_process:
            print(f"✅ AccelerateTrainer 初始化完成")
            print(f"  - 设备: {self.accelerator.device}")
            print(f"  - 进程数: {self.accelerator.num_processes}")
            print(f"  - 混合精度: {self.accelerator.mixed_precision}")
            print(f"  - 梯度累积步数: {hparams.trainer.gradient_accumulation_steps}")
            print(f"  - 总 batch size: {self.total_batch_size}")

    def load_checkpoint(self):
        """加载 checkpoint"""
        if self.hparams.trainer.resume_from is None:
            return
            
        resume_from = self.hparams.trainer.resume_from
        
        if resume_from == "latest":
            checkpoint_dir = self.hparams.trainer.checkpoint_dir
            if os.path.exists(checkpoint_dir):
                checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                    resume_from = os.path.join(checkpoint_dir, checkpoints[-1])
                else:
                    if self.accelerator.is_main_process:
                        print("⚠️ 未找到 checkpoint，从头开始训练")
                    return
            else:
                if self.accelerator.is_main_process:
                    print("⚠️ checkpoint 目录不存在，从头开始训练")
                return
        
        if os.path.exists(resume_from):
            if self.accelerator.is_main_process:
                print(f"📂 正在从 {resume_from} 恢复训练...")
            
            # accelerator.load_state 会自动恢复：模型、优化器、学习率调度器、随机状态
            self.accelerator.load_state(resume_from)
            
            # 从目录名提取 global_step
            self.global_step = int(os.path.basename(resume_from).split("-")[1])
            
            # 加载 EMA（accelerator 不会自动保存 EMA）
            ema_path = os.path.join(resume_from, "ema.pt")
            if self.ema is not None and os.path.exists(ema_path):
                self.ema.load_state_dict(torch.load(ema_path, map_location=self.accelerator.device))
                if self.accelerator.is_main_process:
                    print(f"✅ EMA 模型已从 {ema_path} 加载")
            
            if self.accelerator.is_main_process:
                print(f"✅ 从 step {self.global_step} 恢复训练")
        else:
            if self.accelerator.is_main_process:
                print(f"⚠️ checkpoint 路径 {resume_from} 不存在，从头开始训练")

    def save_checkpoint(self):
        """保存 checkpoint"""
        if self.global_step % self.hparams.trainer.checkpointing_steps != 0:
            return
        
        self.accelerator.wait_for_everyone()
            
        save_path = os.path.join(
            self.hparams.trainer.checkpoint_dir, 
            f"checkpoint-{self.global_step}"
        )
        
        # accelerator.save_state 会自动保存：模型、优化器、学习率调度器、随机状态
        self.accelerator.save_state(save_path)
        
        self.accelerator.wait_for_everyone()
        
        # 保存 EMA 模型（accelerator 不会自动保存 EMA）
        if self.ema is not None and self.accelerator.is_main_process:
            ema_path = os.path.join(save_path, "ema.pt")
            torch.save(self.ema.state_dict(), ema_path)
        
        # 保存模型配置
        if self.accelerator.is_main_process:
            self.accelerator.unwrap_model(self.model).config.save_pretrained(save_path)
            print(f"💾 Checkpoint 已保存到 {save_path}")
        
        self.accelerator.wait_for_everyone()
        
        # 删除旧的 checkpoint（保留 consolidation_steps 的和最近的）
        if self.accelerator.is_main_process:
            checkpoints = [d for d in os.listdir(self.hparams.trainer.checkpoint_dir) 
                          if d.startswith("checkpoint-")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            
            checkpoints_to_keep = set()
            for ckpt in checkpoints:
                step = int(ckpt.split("-")[1])
                if step % self.hparams.trainer.consolidation_steps == 0:
                    checkpoints_to_keep.add(ckpt)
            
            if checkpoints:
                checkpoints_to_keep.add(checkpoints[-1])
            
            for ckpt in checkpoints:
                if ckpt not in checkpoints_to_keep:
                    ckpt_path = os.path.join(self.hparams.trainer.checkpoint_dir, ckpt)
                    shutil.rmtree(ckpt_path)

    def backward(self, loss):
        """反向传播"""
        self.accelerator.backward(loss)

    def optimizer_step(self):
        """优化器更新"""
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), 
                self.hparams.trainer.gradient_clipping
            )
        
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()


def get_trainer(hparams):
    """获取训练器实例"""
    return AccelerateTrainer(hparams)