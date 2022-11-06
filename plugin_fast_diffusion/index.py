from render_utils.plugin import Plugin
from plugin_fast_diffusion.train_dreambooth import train_dreambooth
from typing import List
from PIL import Image

from train_utils import download_image


class FictionFastDiffusion(Plugin):
    hub_token = "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"

    def __init__(
        self,
        user_id: str,
        training_id: str,
        instance_prompt_medium: str = "photo",
        instance_prompt_subject: str = "person",
        instance_prompt_description: str = "young",
    ):
        super().__init__()

        self.user_id = user_id
        self.training_id = training_id
        self.instance_prompt_subject = instance_prompt_subject
        self.instance_prompt = (
            f"a {instance_prompt_medium} of a sxkx {instance_prompt_subject}"
        )

        self.class_prompt = f"a {instance_prompt_medium} of a {instance_prompt_description} sxkx {instance_prompt_subject}"

    def train(self, training_image_urls: List[str] = list()):

        images: List = list(
            filter(None, [download_image(url) for url in training_image_urls])
        )

        save_path = f"/data/train/{self.training_id}"
        instance_data_dir = f"{save_path}/training_data"

        train_dreambooth(
            images=images,
            pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
            hub_token=self.hub_token,
            instance_prompt=self.instance_prompt,
            class_prompt=self.class_prompt,
            with_prior_preservation=False,
            output_dir=f"{save_path}/output",
            instance_data_dir=instance_data_dir,
            class_data_dir=f"{save_path}/class_data",
        )

    def infer(self):
        pass
