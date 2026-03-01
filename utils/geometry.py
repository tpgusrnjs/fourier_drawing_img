import cv2
import numpy as np

def mask_to_contour(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea).squeeze()

def resample_contour(contour, n=256):
    diffs = np.diff(contour, axis=0, append=contour[:1])
    dists = np.linalg.norm(diffs, axis=1)
    arc = np.cumsum(dists)
    arc /= arc[-1]

    t = np.linspace(0, 1, n)
    resampled = np.zeros((n, 2))
    for i, ti in enumerate(t):
        idx = np.searchsorted(arc, ti)
        resampled[i] = contour[idx]
    return resampled