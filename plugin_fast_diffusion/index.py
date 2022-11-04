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
from train_dreambooth import train_dreambooth


class FictionFastDiffusion(Plugin):
    hub_token = "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"
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

    def train(self):
        train(test=1)

    def infer(self):
        pass


handler = FictionFastDiffusion(
    instance_prompt_medium="photo", instance_prompt_subject="man"
)
