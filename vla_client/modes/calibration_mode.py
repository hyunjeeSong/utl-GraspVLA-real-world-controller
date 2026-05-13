"""Interactive data collection for ArUco hand-eye calibration.

Each Enter-press grabs the current camera frames + joint angles and sends a
single ZMQ request to the GraspVLA2 server. The server (with --aruco) detects
the marker and saves a per-frame .json containing T_marker_to_cam (from PnP)
and T_link7_to_base (from FK) into its --aruco-debug-dir.

When you've captured ~7–10 diverse stationary poses, quit (q + Enter) and run

    cd /home/kelly062001/embodiedai/GraspVLA2
    python calibration/solve_marker_calib.py

to compute T_marker_to_link7. Plug the resulting .npy back into the server's
--aruco-t-marker-to-link7 flag for accurate extrinsic.

Pose diversity tips:
    - Vary wrist yaw (q7) and pitch (q6); these change marker orientation.
    - Vary q1 (shoulder yaw) if possible.
    - Keep the marker clearly visible to the side camera.
    - Robot must be STATIONARY at each press (let any motion settle ~1 sec).
"""

from ..utils.cameras import Camera

import controllers
import zmq
import numpy as np
import io
import PIL.Image


class CalibrationMode:
    def __init__(self, args):
        self.args = args
        self.front_camera = Camera(args.front_camera)
        self.side_camera  = Camera(args.side_camera)
        self.robot_controller = controllers.FrankaROSController(
            "logical", args.extented_finger, disable_monitor=True
        )

        self.context = zmq.Context()
        self.socket  = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{args.server_ip}:{args.server_port}")

    def run(self):
        print()
        print("=" * 64)
        print("  ArUco hand-eye CALIBRATION mode")
        print("=" * 64)
        print("Move the robot manually (teach/zero-G or via another tool) to a")
        print("STATIONARY pose with the marker visible in the side camera.")
        print("Press Enter to capture. Type 'q' + Enter to finish.")
        print("Aim for 7-10 poses with diverse wrist rotations.")
        print()

        first = True
        i = 0
        while True:
            line = input(f"[pose {i+1}] Enter to capture (q to quit): ").strip().lower()
            if line == 'q':
                break

            try:
                front_rgb     = self.front_camera.get_frame()
                side_rgb      = self.side_camera.get_frame()
                front_rgb_raw = self.front_camera.get_frame_raw()
                side_rgb_raw  = self.side_camera.get_frame_raw()
                joint_angles_7 = np.asarray(
                    self.robot_controller.latest_qpos, dtype=np.float32
                )[:7]
                eef_pose = self.robot_controller.get_eef_pose()
            except Exception as e:
                print(f"  capture failed: {e}")
                continue

            message = {
                "text": "calibration",
                "front_view_image": [self._compress_image(front_rgb)],
                "side_view_image":  [self._compress_image(side_rgb)],
                "front_view_image_raw": [self._compress_image(front_rgb_raw)],
                "side_view_image_raw":  [self._compress_image(side_rgb_raw)],
                "front_camera_K": np.asarray(self.front_camera.k_real, dtype=np.float64),
                "side_camera_K":  np.asarray(self.side_camera.k_real,  dtype=np.float64),
                "proprio_array":  [eef_pose, eef_pose, eef_pose, eef_pose],
                "joint_angles":   joint_angles_7,
                "reset_ctrnet_buffer": first,
                "use_triangulation": False,
                "compressed": True,
            }
            first = False

            self.socket.send_pyobj(message)
            response = self.socket.recv_pyobj()
            debug = response.get("debug", {}) if isinstance(response, dict) else {}
            front_ok = debug.get("T_front") is not None
            side_ok  = debug.get("T_side")  is not None
            print(f"  q = [{', '.join(f'{x:+.3f}' for x in joint_angles_7)}]")
            print(f"  front camera: {'detected ✓' if front_ok else 'NOT detected'}")
            print(f"  side  camera: {'detected ✓' if side_ok  else 'NOT detected'}")
            i += 1

        print()
        print(f"Captured {i} pose(s). Next step on the server machine:")
        print(f"    cd /home/kelly062001/embodiedai/GraspVLA2")
        print(f"    python calibration/solve_marker_calib.py")
        print()
        print("Then restart the server with:")
        print("    --aruco-t-marker-to-link7 calibration/T_marker_to_link7_calibrated.npy")

    @staticmethod
    def _compress_image(image):
        bytes_io = io.BytesIO()
        PIL.Image.fromarray(image).save(bytes_io, format="JPEG")
        return bytes_io.getvalue()
