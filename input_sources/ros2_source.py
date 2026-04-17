from __future__ import annotations

import threading

import cv2
import numpy as np

from .frame_source import (
    FrameSourceInfo,
    PushFrameSource,
    build_frame_datum,
    build_identity_pose,
    build_pinhole_camera,
    sanitize_sequence_name,
)


class Ros2ImageFrameSource(PushFrameSource):
    def __init__(
        self,
        topic_name: str,
        *,
        compressed: bool = False,
        node_name: str = "boxer_input_source",
        queue_size: int = 8,
        max_frames: int | None = None,
        timeout_sec: float = 1.0,
        resize: tuple[int, int] | None = None,
        camera_name: str = "rgb",
        width: int | None = None,
        height: int | None = None,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        frame_period_ns: int = 100_000_000,
        start_time_ns: int = 0,
        sequence_name: str | None = None,
    ):
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import CompressedImage, Image
        except ImportError as exc:
            raise ImportError(
                "ROS2 input mode requires rclpy and sensor_msgs to be installed in the runtime environment"
            ) from exc

        super().__init__(
            FrameSourceInfo(
                kind="ros2",
                sequence_name=sequence_name or sanitize_sequence_name(topic_name, default="ros2_stream"),
                camera=camera_name,
                device_name="ros2-topic",
                is_live=True,
                metadata={"topic_name": topic_name, "compressed": compressed},
            ),
            resize=resize,
            max_frames=max_frames,
            timeout_sec=timeout_sec,
            queue_size=queue_size,
        )
        self._rclpy = rclpy
        self._Node = Node
        self._Image = Image
        self._CompressedImage = CompressedImage
        self._camera = None
        self._width = width
        self._height = height
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._frame_period_ns = int(frame_period_ns)
        self._start_time_ns = int(start_time_ns)
        self._received = 0
        self._spinning = True
        self._rclpy.init(args=None)
        self._node = self._Node(node_name)
        msg_type = self._CompressedImage if compressed else self._Image
        self._subscription = self._node.create_subscription(
            msg_type,
            topic_name,
            self._on_msg,
            queue_size,
        )
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    def _spin(self):
        while self._spinning and self._rclpy.ok():
            self._rclpy.spin_once(self._node, timeout_sec=0.1)

    def _decode_image(self, msg):
        if isinstance(msg, self._CompressedImage):
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode ROS2 CompressedImage payload")
            return img

        if msg.encoding not in {"rgb8", "bgr8", "mono8"}:
            raise ValueError(f"Unsupported ROS2 Image encoding: {msg.encoding}")
        img = np.frombuffer(msg.data, dtype=np.uint8)
        channels = 1 if msg.encoding == "mono8" else 3
        img = img.reshape(msg.height, msg.width, channels)
        if channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _timestamp_ns(self, msg) -> int:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is not None:
            return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        return self._start_time_ns + self._received * self._frame_period_ns

    def _ensure_camera(self, frame):
        if self._camera is not None:
            return
        frame_h, frame_w = frame.shape[:2]
        self._camera = build_pinhole_camera(
            self._width or frame_w,
            self._height or frame_h,
            fx=self._fx,
            fy=self._fy,
            cx=self._cx,
            cy=self._cy,
        )

    def _on_msg(self, msg):
        try:
            frame = self._decode_image(msg)
            self._ensure_camera(frame)
            datum = build_frame_datum(
                img_bgr=frame,
                timestamp_ns=self._timestamp_ns(msg),
                camera=self._camera,
                pose=build_identity_pose(),
                resize=self.resize,
            )
            self.push_datum(datum)
            self._received += 1
            if self.max_frames is not None and self._received >= self.max_frames:
                self.close()
        except Exception as exc:  # pragma: no cover - best-effort live adapter
            self._node.get_logger().error(f"ROS2 frame ingestion failed: {exc}")
            self.close()

    def close(self) -> None:
        if not self._spinning:
            return
        self._spinning = False
        super().close()
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
