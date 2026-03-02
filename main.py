import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline

from utils.geometry import mask_to_contour, resample_contour
from utils.signal import contour_to_fourier
from src.rendering import render_epicycle_gif

def show_sorted_masks(image, masks):
    img_output_path = "data/output/masks.jpg"

    if not masks:
        print("No objects detected.")
        return

    print(f"Total masks detected: {len(masks)}")

    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    ax = plt.gca()
    ax.set_autoscale_on(False)

    sorted_masks = sorted(
        masks,
        key=lambda x: np.array(x).sum(),
        reverse=True
    )

    for m in sorted_masks:
        m = m.astype(bool)

        color = np.random.rand(3)

        overlay = np.zeros((*m.shape, 4))
        overlay[..., :3] = color
        overlay[..., 3] = m * 0.35

        ax.imshow(overlay)

    plt.axis("off")
    plt.title(f"Detected {len(masks)} objects")

    plt.savefig(img_output_path, dpi=300)
    plt.show()

def main():
    img_path = "data/input.jpg"
    test_img_path = "data/test_input.jpg"

    image = Image.open(img_path).convert("RGB")

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(f"Using device: {device}")

    try:
        print("Downloading model...")
        generator = pipeline("mask-generation", model="facebook/sam-vit-base", device=device)

        print("Generating masks...")
        outputs = generator(
            image,
            pred_iou_thresh=0.98,
            stability_score_thresh=0.96,
            min_mask_region_area=6000,
            #points_per_batch=32
        )

        print(f"visualizing masks...")
        show_sorted_masks(image, outputs["masks"])

        print(f"computing fourier representations...")
        objects = []

        for mask in outputs["masks"]:
            contour = mask_to_contour(mask)
            if contour is None or len(contour) < 100:
                continue

            contour = resample_contour(contour, 256)
            coeffs, freqs, center = contour_to_fourier(contour)
            objects.append((coeffs, freqs, center))

        render_epicycle_gif(
            np.array(image),
            objects,
            save_path="data/output/epicycle_multi_object.gif",
            frames=240,
            K=60
        )
    except Exception as e:
        print(f"Error during mask generation: {e}")

if __name__ == "__main__":
    main()