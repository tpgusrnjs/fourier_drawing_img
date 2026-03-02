import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.geometry import mask_to_contour, resample_contour
from utils.signal import contour_to_fourier


def show_sorted_masks(image, masks, output_path: str = "data/output/masks.jpg"):
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
        reverse=True,
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
    plt.savefig(output_path, dpi=300)
    plt.show()


def masks_to_objects(masks):
    objects = []
    print("masks to objects...")
    for mask in tqdm(masks):
        contour = mask_to_contour(mask)
        if contour is None or len(contour) < 100:
            continue

        contour = resample_contour(contour, 256)
        coeffs, freqs, center = contour_to_fourier(contour)
        objects.append((coeffs, freqs, center))
    return objects
