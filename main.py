from train_utils import download_image
from typing import List

from plugin_fast_diffusion.index import FictionFastDiffusion


training_image_urls = list(
    [
        "https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg",
        "https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg",
        "https://huggingface.co/datasets/valhalla/images/resolve/main/5.jpeg",
        "https://huggingface.co/datasets/valhalla/images/resolve/main/6.jpeg",
    ]
)

images: List = list(filter(None, [download_image(url) for url in training_image_urls]))

instance_prompt_medium = "photo"
instance_prompt_subject = "man"
instance_prompt_description = "rich"
user_id = "user1234"
training_id = "train1234"

handler = FictionFastDiffusion(
    instance_prompt_medium=instance_prompt_medium,
    instance_prompt_subject=instance_prompt_subject,
    instance_prompt_description=instance_prompt_description,
    user_id=user_id,
    training_id=training_id,
)

handler.train(training_image_urls=training_image_urls)
