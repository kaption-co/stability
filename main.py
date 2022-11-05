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

instance_prompt = "a photo of a sxkx person"

handler = FictionFastDiffusion(instance_prompt)

handler.train(training_image_urls)
