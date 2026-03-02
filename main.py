import os

import numpy as np
from PIL import Image

######################################################################################################################
# Workaround for argparse help validation issue with Hydra (see https://github.com/facebookresearch/hydra/issues/3121)
# I recommend to use Python 3.13 until Hydra adds full 3.14 support.
import argparse
_orig_check_help = argparse.ArgumentParser._check_help

def _patched_check_help(self, action):
    if not isinstance(action.help, str):
        try:
            action.help = str(action.help)
        except Exception:
            action.help = ""
    return _orig_check_help(self, action)

argparse.ArgumentParser._check_help = _patched_check_help
######################################################################################################################

import hydra
from omegaconf import DictConfig

from src import model, segmentation
from src.rendering import render_epicycle_gif


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    img_path = cfg.img_path
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    model_name = cfg.model_name

    image = Image.open(img_path).convert("RGB")

    generator = model.setup_model(model_name)

    try:
        outputs = model.predict_masks(generator, image, cfg.amg)

        print("visualizing masks...")
        segmentation.show_sorted_masks(image, outputs.get("masks", []), output_path=f"data/output/masks_{img_name}.jpg")

        print("computing fourier representations...")
        objects = segmentation.masks_to_objects(outputs.get("masks", []))

        render_epicycle_gif(
            np.array(image),
            objects,
            save_path=f"data/output/fourier_drawing_{img_name}.gif",
            frames=240,
            K=60
        )
    except Exception as e:
        print(f"Error during mask generation: {e}")

if __name__ == "__main__":
    main()