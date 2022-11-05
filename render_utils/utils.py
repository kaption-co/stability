import requests
from io import BytesIO
from PIL import Image


def download_image(url: str) -> Image:
    try:
        response = requests.get(url)
    except:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")
