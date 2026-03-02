import torch
from transformers import pipeline


def setup_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Downloading model...")
    return pipeline("mask-generation", model=model_name, device=device)


def predict_masks(
    generator,
    image,
    cfg
):
    print("Generating masks...")
    return generator(
        image,
        points_per_side=cfg.points_per_side,
        pred_iou_thresh=cfg.pred_iou_thresh,
        stability_score_thresh=cfg.stability_score_thresh,
        stability_score_offset=cfg.stability_score_offset,
        box_nms_thresh=cfg.box_nms_thresh,
        crop_n_layers=cfg.crop_n_layers,
        crop_nms_thresh=cfg.crop_nms_thresh,
        crop_overlap_ratio=cfg.crop_overlap_ratio,
        crop_n_points_downscale_factor=cfg.crop_n_points_downscale_factor,
        min_mask_region_area=cfg.min_mask_region_area,
    )
