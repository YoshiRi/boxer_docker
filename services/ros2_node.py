#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

import numpy as np

from boxer_api import BoxerConfig, BoxerInferenceEngine, BoxerInferenceRequest, DetectorConfig, FrameInput
from input_sources.frame_source import build_identity_pose, build_pinhole_camera
from services.ros2_messages import (
    frame_result_to_detection2d_array,
    frame_result_to_detection3d_array,
    frame_result_to_track3d_array,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal ROS2 node wrapper around boxer_api."
    )
    parser.add_argument("--input-topic", required=True, help="ROS2 image topic name.")
    parser.add_argument(
        "--output-topic",
        default="/boxer/detections_json",
        help="ROS2 std_msgs/String topic for serialized detections.",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Subscribe to sensor_msgs/CompressedImage instead of sensor_msgs/Image.",
    )
    parser.add_argument("--node-name", default="boxer_api_node")
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fx", type=float, default=None)
    parser.add_argument("--camera-fy", type=float, default=None)
    parser.add_argument("--camera-cx", type=float, default=None)
    parser.add_argument("--camera-cy", type=float, default=None)
    parser.add_argument(
        "--labels",
        type=str,
        default="lvisplus",
        help="Comma-separated detection label prompts.",
    )
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--force-precision", choices=["float32", "bfloat16"], default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--thresh2d", type=float, default=0.25)
    parser.add_argument("--thresh3d", type=float, default=0.5)
    return parser


class BoxerRos2Node:
    def __init__(self, args) -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import CompressedImage, Image
            from std_msgs.msg import String
        except ImportError as exc:  # pragma: no cover - runtime environment dependent
            raise ImportError(
                "services/ros2_node.py requires rclpy, sensor_msgs, and std_msgs"
            ) from exc

        import cv2

        self._cv2 = cv2
        self._rclpy = rclpy
        self._Image = Image
        self._CompressedImage = CompressedImage
        self._String = String
        self._rclpy.init(args=None)
        self.node = Node(args.node_name)
        self._args = args
        self._camera = None
        labels = [item.strip() for item in args.labels.split(",") if item.strip()]
        self._engine = BoxerInferenceEngine(
            detector=DetectorConfig(
                labels=labels or ["lvisplus"],
                threshold_2d=args.thresh2d,
                force_precision=args.force_precision,
            ),
            boxer=BoxerConfig(
                threshold_3d=args.thresh3d,
                checkpoint_path=args.ckpt,
                force_cpu=args.force_cpu,
                force_precision=args.force_precision,
            ),
        )
        self._publisher = self.node.create_publisher(String, args.output_topic, args.queue_size)
        msg_type = CompressedImage if args.compressed else Image
        self._subscription = self.node.create_subscription(
            msg_type,
            args.input_topic,
            self._on_msg,
            args.queue_size,
        )

    def spin(self) -> None:
        self.node.get_logger().info(
            f"Subscribing to {self._args.input_topic}, publishing to {self._args.output_topic}"
        )
        self._rclpy.spin(self.node)

    def close(self) -> None:
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()

    def _decode_image(self, msg):
        if isinstance(msg, self._CompressedImage):
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = self._cv2.imdecode(arr, self._cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode CompressedImage")
            return img

        if msg.encoding not in {"rgb8", "bgr8", "mono8"}:
            raise ValueError(f"Unsupported ROS2 Image encoding: {msg.encoding}")
        img = np.frombuffer(msg.data, dtype=np.uint8)
        channels = 1 if msg.encoding == "mono8" else 3
        img = img.reshape(msg.height, msg.width, channels)
        if channels == 1:
            img = self._cv2.cvtColor(img, self._cv2.COLOR_GRAY2BGR)
        elif msg.encoding == "rgb8":
            img = self._cv2.cvtColor(img, self._cv2.COLOR_RGB2BGR)
        return img

    def _timestamp_ns(self, msg) -> int:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None:
            return 0
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _ensure_camera(self, image_bgr: np.ndarray):
        if self._camera is not None:
            return self._camera
        height, width = image_bgr.shape[:2]
        self._camera = build_pinhole_camera(
            self._args.camera_width or width,
            self._args.camera_height or height,
            fx=self._args.camera_fx,
            fy=self._args.camera_fy,
            cx=self._args.camera_cx,
            cy=self._args.camera_cy,
        )
        return self._camera

    def _on_msg(self, msg) -> None:  # pragma: no cover - runtime environment dependent
        try:
            image_bgr = self._decode_image(msg)
            frame = FrameInput(
                image_bgr=image_bgr,
                camera=self._ensure_camera(image_bgr),
                pose_world_rig=build_identity_pose(),
                sparse_points_world=np.zeros((0, 3), dtype=np.float32),
                timestamp_ns=self._timestamp_ns(msg),
                rotated=False,
                source_name=self._args.input_topic,
                device_name="ros2-topic",
            )
            result = self._engine.infer_frame(
                BoxerInferenceRequest(
                    frame=frame,
                )
            )
            payload = self._serialize_result(result)
            ros_msg = self._String()
            ros_msg.data = json.dumps(payload, separators=(",", ":"))
            self._publisher.publish(ros_msg)
        except Exception as exc:
            self.node.get_logger().error(f"boxer_api ROS2 callback failed: {exc}")

    def _serialize_result(self, result) -> dict:
        return {
            "detection_2d_array": frame_result_to_detection2d_array(
                result,
                frame_id=self._args.input_topic,
            ),
            "detection_3d_array": frame_result_to_detection3d_array(
                result,
                frame_id=self._args.input_topic,
            ),
            "track_3d_array": frame_result_to_track3d_array(
                result,
                frame_id=self._args.input_topic,
            ),
            "timings_ms": result.timings_ms,
            "metadata": result.metadata,
        }


def main(argv=None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    node = BoxerRos2Node(args)
    try:
        node.spin()
    finally:
        node.close()


if __name__ == "__main__":
    main()
