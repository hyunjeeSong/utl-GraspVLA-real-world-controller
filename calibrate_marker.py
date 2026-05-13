"""Entry point for interactive ArUco hand-eye calibration data collection.

Usage:
python3 calibrate_marker.py \
    --front-camera 243622072408 --side-camera 348522071378 \
    --server-ip 163.152.162.236 --server-port 6667

Same args as main.py minus the action-related ones. The server side must be
running with --aruco --aruco-debug-dir <path> so each capture saves a .json.
"""

import argparse
import sys

import rospy

sys.path.append('./vla_client')
from vla_client.modes import CalibrationMode


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--front-camera", type=str, required=True,
                        help="front camera serial number")
arg_parser.add_argument("--side-camera", type=str, required=True,
                        help="side camera serial number")
arg_parser.add_argument("--server-ip", type=str, required=True)
arg_parser.add_argument("--server-port", type=int, required=True)
arg_parser.add_argument("--extented_finger", action='store_true',
                        help="use the extended finger configuration")


if __name__ == "__main__":
    args = arg_parser.parse_args()

    import pyrealsense2
    ctx = pyrealsense2.context()
    print("RealSense devices:")
    for i, dev in enumerate(ctx.query_devices()):
        print(f"  [{i}] {dev}")

    rospy.init_node('vla_calibrator', anonymous=True, disable_signals=True)
    CalibrationMode(args).run()
    rospy.signal_shutdown('done')
