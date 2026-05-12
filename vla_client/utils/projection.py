"""
Pinhole projection helpers — convert a 3D point in robot-base frame to a 2D
pixel coordinate, using an extrinsic (T_base_to_cam) returned by the server's
CTRNet-X estimator and a camera intrinsic K.

K may need transformation if the displayed image is a cropped+resized variant
of the raw camera frame (e.g. cameras.py crops 640×480 → 480×480 then resizes
to 256×256). transform_K_for_crop_resize() applies that same transform to K
so the pixel lands in the displayed image's coordinate space.
"""
import numpy as np


def project_base_to_pixel(xyz_base, T_base_to_cam, K):
    """3D point in robot-base frame → 2D image pixel.

    xyz_base       : (3,) point in robot base frame [meters]
    T_base_to_cam  : (4,4) extrinsic from server (debug['T_front'/'T_side'])
    K              : (3,3) camera intrinsic matching the target image space

    Returns (u, v) tuple of floats, or None if point is behind the camera.
    """
    if xyz_base is None or T_base_to_cam is None or K is None:
        return None
    xyz_h = np.append(np.asarray(xyz_base, dtype=np.float64), 1.0)  # (4,)
    cam = np.asarray(T_base_to_cam, dtype=np.float64)[:3, :] @ xyz_h  # (3,)
    if cam[2] <= 0.01:   # behind camera or too close
        return None
    px = np.asarray(K, dtype=np.float64) @ cam   # (3,)
    return float(px[0] / px[2]), float(px[1] / px[2])


def transform_K_for_crop_resize(K_orig, crop_offset, resize_scale):
    """Apply center-crop + uniform resize to K, matching cameras.py:crop_frame.

    For RealSense 640×480 → center crop 480×480 (offset (80, 0)) → resize 256×256:
        K_256 = transform_K_for_crop_resize(K_real, crop_offset=(80, 0),
                                            resize_scale=256/480)

    crop_offset  : (offset_x, offset_y) pixels removed from top-left
    resize_scale : uniform scale applied after crop (e.g. 256/480)
    """
    K = np.asarray(K_orig, dtype=np.float64).copy()
    K[0, 2] -= crop_offset[0]
    K[1, 2] -= crop_offset[1]
    K[:2, :] *= resize_scale
    return K
