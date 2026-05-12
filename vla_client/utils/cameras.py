import cv2
from .realsense import RealSenseCamera
from collections import OrderedDict
import rospy
import threading
from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge
import numpy as np


class Camera:
    # RealSense RGB stream resolution. 1280×720 quadruples pixel count vs the
    # previous 640×480, so a 50 mm marker at 1 m goes from ~30 px to ~60 px
    # per side — ArUco corner sub-pixel refinement is much more accurate, and
    # the same K_real (from RealSense SDK) is automatically updated for the
    # new resolution. Effective FOV of the 256×256 VLA input after the
    # center-crop+resize stays ≈43°, matching the VLA training distribution.
    CAMERA_WIDTH  = 1280
    CAMERA_HEIGHT = 720

    def __init__(self, serial_number):
        self.camera = RealSenseCamera(serial_number, fps=30,
                                       width=self.CAMERA_WIDTH,
                                       height=self.CAMERA_HEIGHT)
        self.k_real = self.camera.get_camera_intrinsics_matrix()
        print(f'camera {self.CAMERA_WIDTH}x{self.CAMERA_HEIGHT} intrinsics {self.k_real}')

    def get_frame(self):
        rgb, depth= self.camera.get_frames()
        rgb = self.crop_frame(rgb)
        return rgb

    def get_frame_raw(self):
        """Return native RealSense RGB frame (CAMERA_WIDTH × CAMERA_HEIGHT)
        without crop/resize. Used for CTRNet / ArUco extrinsic estimation —
        keeping native resolution + aspect ratio avoids the vertical squash
        that 256×256 → 320×240 introduces, and self.k_real already matches
        this resolution.
        """
        rgb, _ = self.camera.get_frames()
        return rgb

    def crop_frame(self, image):
        """Generic center-square crop + resize to 256×256 (VLA input space).
        Works for any RealSense resolution: takes the central min(H,W) square,
        then resamples down/up to 256×256.
        """
        H, W, _ = image.shape
        s = min(H, W)
        start_x = (W - s) // 2
        start_y = (H - s) // 2
        new_image = image[start_y:start_y + s, start_x:start_x + s, :]
        res = cv2.resize(new_image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        return res

    def get_K_256(self):
        """Intrinsic matrix for the 256×256 view produced by crop_frame().

        Applies the same center-crop + resize transform to self.k_real:
        principal point shifts by the crop offset; fx/fy/cx/cy then scale by
        256 / min(H, W). Use this when re-projecting a base-frame 3D point onto
        the visualizer image (which displays the 256×256 stream resized to
        self.size). Independent of CAMERA_WIDTH / CAMERA_HEIGHT so it stays
        correct if the camera resolution changes.
        """
        H, W = self.CAMERA_HEIGHT, self.CAMERA_WIDTH
        s = min(H, W)
        crop_x = (W - s) // 2
        crop_y = (H - s) // 2
        scale = 256.0 / s
        K = np.asarray(self.k_real, dtype=np.float64).copy()
        K[0, 2] -= crop_x
        K[1, 2] -= crop_y
        K[:2, :] *= scale
        return K


class CameraVisualizer:
    def __init__(self, cameras: OrderedDict, hz=10, size=(512, 512), ref_images=None):
        self.cameras = cameras
        self.cv_bridge = CvBridge()

        self.hz = hz
        self.size = size

        if ref_images is not None:
            assert len(ref_images) == len(cameras), 'Number of ref images must be equal to number of cameras'
            self.ref_images = []
            for ref_image_path in ref_images:
                ref_image = cv2.resize(cv2.imread(ref_image_path), size)
                robot_mask = np.sum(ref_image, axis=-1) > 0
                ref_image[robot_mask, 0] = 0
                ref_image[robot_mask, 1] = 255
                ref_image[robot_mask, 2] = 0
                self.ref_images.append(ref_image)
        else:
            self.ref_images = None

        self.publishers = {}
        self.bboxes = {}
        # Goal pixels overlaid alongside bbox. Each entry is a dict mapping a
        # color label ("final" / "vlm") to ((u, v), reference_image_size).
        # "final" = post-triangulation goal (debug['pose']) — drawn red
        # "vlm"   = raw VLM goal (debug['pose_raw'])         — drawn blue
        self.goal_pixels = {}
        for name in cameras:
            self.publishers[name] = rospy.Publisher(f'/cameras/{name}', ImageMsg, queue_size=0)
            if self.ref_images:
                self.publishers[f"{name}_ref"] = rospy.Publisher(f'/cameras/{name}_ref', ImageMsg, queue_size=0)

        def fn():
            rate = rospy.Rate(self.hz)
            while not rospy.is_shutdown():
                self._step()
                rate.sleep()

        self.step_thread = threading.Thread(target=fn, daemon=True)
        self.step_thread.start()

    def _step(self):
        frames = []
        for idx, (name, camera) in enumerate(self.cameras.items()):
            try:
                color_frame = camera.get_frame()
                color_frame = cv2.cvtColor(cv2.resize(color_frame.copy(), self.size), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f'Error getting frame from camera {name}: {e}')
                break

            if name in self.bboxes:
                color_frame = self.draw_bbox(color_frame, *self.bboxes[name])
            if name in self.goal_pixels:
                color_frame = self.draw_goal_pixels(color_frame, self.goal_pixels[name])
            self.draw_center_cross(color_frame)

            if self.ref_images is not None:
                diff = (color_frame * 0.8 + self.ref_images[idx] * 0.2).astype('uint8')
                try:
                    self.publishers[name + '_ref'].publish(self.cv_bridge.cv2_to_imgmsg(diff, "bgr8"))
                except:
                    pass

            frames.append(color_frame)
            try:
                self.publishers[name].publish(self.cv_bridge.cv2_to_imgmsg(color_frame, "bgr8"))
            except:
                raise

    def draw_center_cross(self, frame):
        cross_length = 50
        cv2.line(frame, (frame.shape[0]//2-cross_length, frame.shape[1]//2), (frame.shape[0]//2+cross_length, frame.shape[1]//2), color=(0, 255, 0), thickness=1)
        cv2.line(frame, (frame.shape[0]//2, frame.shape[1]//2-cross_length), (frame.shape[0]//2, frame.shape[1]//2+cross_length), color=(0, 255, 0), thickness=1)

    def draw_bbox(self, frame, bbox, image_size):
        original_size = frame.shape[:2]
        # directly resize the bbox
        bbox = np.array(bbox)
        image_size = np.array(image_size)
        bbox = (bbox * np.array([*(original_size/image_size)]*2)).astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        return frame
    
    def set_bbox(self, camera_name, bbox):
        self.bboxes[camera_name] = bbox

    def set_goal_pixels(self, camera_name, pixels_dict):
        """pixels_dict = {label: ((u, v), reference_image_size)} per camera.
        label ∈ {'final', 'vlm'} — final is red, vlm is blue.
        Pass an empty dict to clear without removing the camera entry."""
        self.goal_pixels[camera_name] = pixels_dict

    def draw_goal_pixels(self, frame, pixels_dict):
        """Draw goal points on `frame` (BGR uint8). pixels_dict per set_goal_pixels.

        Draw order: 'vlm' (blue, raw VLM) FIRST, 'final' (red, applied goal) LAST.
        Final is the goal actually sent to the action expert, so it stays on top
        even when triangulation is rejected and both points coincide.
        """
        H_disp, W_disp = frame.shape[:2]
        colors = {
            'final': (0,   0, 255),   # BGR red  — post-triangulation (or VLM if rejected)
            'vlm':   (255, 100,  0),  # BGR blue — raw VLM goal (pre-triangulation)
        }
        # Iterate in explicit order: vlm first (so final is drawn on top, always visible).
        for label in ('vlm', 'final'):
            value = pixels_dict.get(label)
            if value is None:
                continue
            (u, v), (H_ref, W_ref) = value
            u_disp = int(round(u * (W_disp / W_ref)))
            v_disp = int(round(v * (H_disp / H_ref)))
            color  = colors.get(label, (255, 255, 255))
            cv2.circle(frame, (u_disp, v_disp), 8, color, -1)
            cv2.circle(frame, (u_disp, v_disp), 12, (255, 255, 255), 2)
        return frame

    def clear(self):
        self.bboxes.clear()
        self.goal_pixels.clear()