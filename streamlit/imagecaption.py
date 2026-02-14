# imagecaption.py
from __future__ import annotations

from typing import Optional
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


@torch.inference_mode()
def generate_caption(
    image: Image.Image,
    model_name: str = "Salesforce/blip-image-captioning-base",
    device: Optional[str] = None,
    max_new_tokens: int = 40,
) -> str:
    """
    Generate an image caption using BLIP.

    Requirements:
    - This function is in a separate module (imagecaption.py) as required.
    - Uses Hugging Face BLIP (BLIP-1) for captioning.

    Notes:
    - Works on CPU by default.
    - If you have GPU: device="cuda"
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(out_ids[0], skip_special_tokens=True)
    return caption.strip()
