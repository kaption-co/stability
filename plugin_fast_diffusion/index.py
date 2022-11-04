# @title Import required libraries
import argparse
import itertools
import math
import os
from contextlib import nullcontext
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from pathlib import Path
import bitsandbytes as bnb
from train_utils import image_grid, download_image
import math
import gc
from utils.plugin import Plugin


class FictionFastDiffusion(Plugin):
    hf_token = "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"
    name = "Fast Diffusion 1.5"
    resolution = (512,)
    center_crop = (True,)
    learning_rate = (5e-06,)
    max_train_steps = (450,)
    train_batch_size = (1,)
    gradient_accumulation_steps = (2,)
    max_grad_norm = (1.0,)
    mixed_precision = ("no",)  # set to "fp16" for mixed-precision training.
    gradient_checkpointing = (True,)  # set this to True to lower the memory usage.
    use_8bit_adam = (True,)  # use 8bit optimizer from bitsandbytes
    seed = (3434554,)
    sample_batch_size = (2,)
    output_dir = "./rendered_concepts"
    base_model = f"runwayml/stable-diffusion-v1-5"

    instance_data_dir = "./training_images"
    instance_prompt = (instance_prompt,)

    with_prior_preservation = False
    prior_loss_weight = 0.5

    class_data_dir = "./class_images"

    num_class_images = 8

    def __init__(
        self,
        instance_prompt_medium: str = "photo",
        instance_prompt_subject: str = "person",
        subject_description: str = "young",
    ):
        super().__init__()
        self.instance_prompt = (
            f"a {instance_prompt_medium} of a sxkx {instance_prompt_subject}"
        )

        self.class_prompt = f"a {instance_prompt_medium} of a {subject_description} sxkx {instance_prompt_subject}"

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.base_model, subfolder="text_encoder", use_auth_token=self.hf_token
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.base_model, subfolder="vae", use_auth_token=self.hf_token
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.base_model, subfolder="unet", use_auth_token=self.hf_token
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.base_model, subfolder="tokenizer", use_auth_token=self.hf_token
        )

    def train(self, images=None, labels=None):
        logger = get_logger(__name__)

        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
        )

        set_seed(self.seed)

        if self.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if self.use_8bit_adam:
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            unet.parameters(),  # only optimize unet
            lr=self.learning_rate,
        )

        noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )

        train_dataset = DreamBoothDataset(
            instance_data_root=self.instance_data_dir,
            instance_prompt=self.instance_prompt,
            class_data_root=self.class_data_dir
            if self.with_prior_preservation
            else None,
            class_prompt=self.class_prompt,
            tokenizer=self.tokenizer,
            size=self.resolution,
            center_crop=self.center_crop,
        )

        def collate_fn(examples):
            input_ids = [example["instance_prompt_ids"] for example in examples]
            pixel_values = [example["instance_images"] for example in examples]

            # concat class and instance examples for prior preservation
            if self.with_prior_preservation:
                input_ids += [example["class_prompt_ids"] for example in examples]
                pixel_values += [example["class_images"] for example in examples]

            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format
            ).float()

            input_ids = self.tokenizer.pad(
                {"input_ids": input_ids}, padding=True, return_tensors="pt"
            ).input_ids

            batch = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
            }
            return batch

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        unet, optimizer, train_dataloader = accelerator.prepare(
            unet, optimizer, train_dataloader
        )

        # Move text_encode and vae to gpu
        self.text_encoder.to(accelerator.device)
        self.vae.to(accelerator.device)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.gradient_accumulation_steps
        )
        num_train_epochs = math.ceil(self.max_train_steps / num_update_steps_per_epoch)

        # Train!
        total_batch_size = (
            self.train_batch_size
            * accelerator.num_processes
            * self.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(self.max_train_steps), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description("Steps")
        global_step = 0

        for epoch in range(num_train_epochs):
            unet.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    with torch.no_grad():
                        latents = self.vae.encode(
                            batch["pixel_values"]
                        ).latent_dist.sample()
                        latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn(latents.shape).to(latents.device)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    ).long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    with torch.no_grad():
                        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample

                    if self.with_prior_preservation:
                        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                        noise, noise_prior = torch.chunk(noise, 2, dim=0)

                        # Compute instance loss
                        loss = (
                            F.mse_loss(noise_pred, noise, reduction="none")
                            .mean([1, 2, 3])
                            .mean()
                        )

                        # Compute prior loss
                        prior_loss = (
                            F.mse_loss(noise_pred_prior, noise_prior, reduction="none")
                            .mean([1, 2, 3])
                            .mean()
                        )

                        # Add the prior loss to the instance loss.
                        loss = loss + self.prior_loss_weight * prior_loss
                    else:
                        loss = (
                            F.mse_loss(noise_pred, noise, reduction="none")
                            .mean([1, 2, 3])
                            .mean()
                        )

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            unet.parameters(), self.max_grad_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)

                if global_step >= self.max_train_steps:
                    break

            accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline(
                text_encoder=self.text_encoder,
                vae=self.vae,
                unet=accelerator.unwrap_model(unet),
                tokenizer=self.tokenizer,
                scheduler=PNDMScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    skip_prk_steps=True,
                ),
                safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                    "CompVis/stable-diffusion-safety-checker"
                ),
                feature_extractor=CLIPFeatureExtractor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                ),
            )
            pipeline.save_pretrained(self.output_dir)

    def infer(self):
        pass


handler = FictionFastDiffusion(
    instance_prompt_medium="photo", instance_prompt_subject="man"
)
