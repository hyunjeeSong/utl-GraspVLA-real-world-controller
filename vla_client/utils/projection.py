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
import transforms3d as t3d


# 5 gripper landmark points in EEF frame, matching playground's
# project_pose_to_pixels (W=0.08 finger separation, D=0.04 finger depth along +z,
# L=0.06 wrist length along -z). Order: palm_L, palm_R, tip_L, tip_R, wrist_top.
_GRIPPER_PTS_EEF = np.array([
    [0, -0.04, -0.04],
    [0,  0.04, -0.04],
    [0, -0.04,  0.00],
    [0,  0.04,  0.00],
    [0,  0.00, -0.10],
], dtype=np.float64)


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


def project_gripper_to_pixels(xyz_base, rpy, T_base_to_cam, K):
    """Project the 5-point gripper stick-figure from a goal pose to image pixels.

    xyz_base       : (3,) goal position in robot base frame [meters]
    rpy            : (3,) goal orientation in 'sxyz' Euler convention [rad]
    T_base_to_cam  : (4,4) extrinsic from server
    K              : (3,3) intrinsic matching the target display image

    Returns: list of 5 (u, v) float tuples
        [palm_L, palm_R, tip_L, tip_R, wrist_top]
    Returns None if T or K missing, or if any landmark is behind the camera
    (we drop the whole gripper instead of half-drawing it).
    """
    if xyz_base is None or rpy is None or T_base_to_cam is None or K is None:
        return None
    rot = t3d.euler.euler2mat(*np.asarray(rpy, dtype=np.float64))
    pts_base = np.asarray(xyz_base, dtype=np.float64) + (rot @ _GRIPPER_PTS_EEF.T).T  # (5, 3)
    pts_h = np.hstack([pts_base, np.ones((5, 1))])                                    # (5, 4)
    cam = (np.asarray(T_base_to_cam, dtype=np.float64)[:3, :] @ pts_h.T).T            # (5, 3)
    if (cam[:, 2] <= 0.01).any():
        return None
    proj = (np.asarray(K, dtype=np.float64) @ cam.T).T                                # (5, 3)
    return [(float(p[0] / p[2]), float(p[1] / p[2])) for p in proj]


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
