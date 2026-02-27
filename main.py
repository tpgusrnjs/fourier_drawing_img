import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline


def show_sorted_masks(image, masks):
    if not masks:
        print("No objects detected.")
        return
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    ax = plt.gca()
    ax.set_autoscale_on(False)

    sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)

    for m in sorted_masks:
        m = m["segmentation"]

        color = np.random.rand(3)

        overlay = np.zeros((*m.shape, 4))
        overlay[..., :3] = color
        overlay[..., 3] = m * 0.35

        ax.imshow(overlay)
    
    plt.axis("off")
    plt.title(f"Detected {len(masks)} objects")
    plt.show()

def main():
    img_path = "input.jpg"
    image = Image.open(img_path).convert("RGB")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        generator = pipeline("mask-generation", model="facebook/sam-vit-base", device=device)
        outputs = generator(image)
        show_sorted_masks(image, outputs['masks'])
    except Exception as e:
        print(f"Error during mask generation: {e}")