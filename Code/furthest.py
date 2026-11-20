import cv2
import numpy as np

def furthest(mask, scan=0, pixel_threshold=50, debug=False):
    """Return (column, row) with maximum clearance in the largest gap.

    Preserve the existing largest-component rule, but measure distance to ALL
    background pixels, including islands inside a gap. Return None for an
    empty/too-small mask. The returned coordinates are pixel indices; callers
    add 0.5 when constructing a camera ray through the pixel centre.
    """
    if mask is None:
        return None
    values = np.asarray(mask)
    if values.ndim != 2 or values.size == 0:
        return None
    img = (np.isfinite(values) & (values > 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    if n <= 1:
        return None
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[label, cv2.CC_STAT_AREA] <= pixel_threshold:
        return None
    component = (labels == label).astype(np.uint8)
    # Padding makes the image boundary count as a boundary too. Crop it back
    # before finding the maximum, so no coordinate offset is introduced.
    distance = cv2.distanceTransform(np.pad(component, 1), cv2.DIST_L2,
                                     cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    _, clearance, _, point = cv2.minMaxLoc(distance)
    if clearance <= 0:
        return None
    if debug:
        cv2.imwrite("Init_Mask_scan_%s.png" % scan, img * 255)
        overlay = cv2.cvtColor(component * 255, cv2.COLOR_GRAY2BGR)
        cv2.circle(overlay, point, 5, (255, 0, 0), -1)
        cv2.imwrite("scan_%s_mask.png" % scan, overlay)
    return point
