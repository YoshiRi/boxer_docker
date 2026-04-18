#!/usr/bin/env python3
"""End-to-end sensor demo: sensor input → 3D bbox → visualisation.

Automatically selects the input source:
  1. Intel RealSense D4xx  (if pyrealsense2 is installed and a device is connected)
  2. Webcam / video file  (--input argument, default: webcam 0)

Detector is selectable via --detector:
  "owl"                               — built-in OWLv2 (no extra deps)
  "IDEA-Research/grounding-dino-base" — Grounding DINO from HuggingFace
  "google/owlv2-base-patch16-ensemble"— OWLv2 from HuggingFace

Examples
--------
    # RealSense + Grounding DINO, detect furniture
    python demo_sensor.py \\
        --detector IDEA-Research/grounding-dino-base \\
        --labels "chair,table,monitor,sofa"

    # Webcam + built-in OWL, full LVIS taxonomy
    python demo_sensor.py --input 0 --detector owl

    # Video file + OWLv2 HF
    python demo_sensor.py \\
        --input my_video.mp4 \\
        --detector google/owlv2-base-patch16-ensemble \\
        --labels "person,backpack,laptop"
"""
import argparse
import sys
import time


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Boxer sensor demo")
    p.add_argument("--input", type=str, default="0",
                   help="webcam index, video path, or 'realsense' (default: 0)")
    p.add_argument("--detector", type=str, default="owl",
                   help="'owl' or a HuggingFace model ID")
    p.add_argument("--labels", type=str, default="lvisplus",
                   help="comma-separated labels or taxonomy name (default: lvisplus)")
    p.add_argument("--thresh2d", type=float, default=0.25)
    p.add_argument("--thresh3d", type=float, default=0.5)
    p.add_argument("--track", action="store_true",
                   help="Enable online 3D box tracking")
    p.add_argument("--skip_viz", action="store_true",
                   help="Disable per-frame visualisation (faster)")
    p.add_argument("--output_dir", type=str, default="output/")
    p.add_argument("--ckpt", type=str,
                   default="ckpts/boxernet_hw960in4x6d768-wssxpf9p.ckpt")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--width", type=int, default=640,
                   help="Camera width for RealSense (default: 640)")
    p.add_argument("--height", type=int, default=480,
                   help="Camera height for RealSense (default: 480)")
    p.add_argument("--fps", type=int, default=30,
                   help="Frame rate for RealSense (default: 30)")
    p.add_argument("--show_viz", action="store_true",
                   help="Display visualisation frames in an OpenCV window")
    return p


def _labels_from_str(s: str) -> list[str]:
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]  # single token or taxonomy name


# ---------------------------------------------------------------------------
# Result handler
# ---------------------------------------------------------------------------

_last_result_time: float = 0.0
_frame_count: int = 0


def make_result_handler(show_viz: bool):
    import threading
    _lock = threading.Lock()
    _counters = {"frames": 0, "detections": 0}

    def handle_result(result) -> None:
        with _lock:
            _counters["frames"] += 1
            _counters["detections"] += len(result.detections)
            fc = _counters["frames"]

        ts_s = result.timestamp_ns / 1e9
        dets = result.detections

        # Console output (every frame)
        det_strs = [f"{d.label}({d.confidence:.2f})" for d in dets]
        print(f"[frame {fc:4d} | t={ts_s:.3f}s] {len(dets)} 3D boxes: {', '.join(det_strs) or '—'}")

        # Optional OpenCV display
        if show_viz and result.viz_jpg:
            import cv2
            import numpy as np
            buf = np.frombuffer(result.viz_jpg, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow("Boxer 3D", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

    return handle_result


# ---------------------------------------------------------------------------
# Source drivers
# ---------------------------------------------------------------------------

def _run_realsense(args, pipeline) -> None:
    from input_sources.realsense_source import RealSenseSource
    src = RealSenseSource(
        pipeline,
        width=args.width,
        height=args.height,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    src.run()


def _run_opencv(args, pipeline) -> None:
    """Webcam / video fallback using OpenCV."""
    import cv2
    import numpy as np
    from boxer_pipeline import Intrinsics, SensorFrame

    src = args.input
    # Webcam index or video path
    cap_src = int(src) if src.isdigit() else src
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print(f"[demo_sensor] Cannot open capture source: {src}", file=sys.stderr)
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    focal = float(max(W, H))
    intrinsics = Intrinsics(width=W, height=H, fx=focal, fy=focal)

    print(f"[demo_sensor] OpenCV source {src} — {W}×{H}")
    frame_count = 0
    try:
        while args.max_frames is None or frame_count < args.max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = bgr[:, :, ::-1].copy()
            frame = SensorFrame(
                rgb=rgb,
                intrinsics=intrinsics,
                timestamp_ns=time.time_ns(),
            )
            pipeline.push(frame)
            frame_count += 1
    except KeyboardInterrupt:
        print("[demo_sensor] interrupted")
    finally:
        cap.release()
        pipeline.stop()
        if args.show_viz:
            import cv2 as cv
            cv.destroyAllWindows()


def _try_realsense() -> bool:
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        return len(ctx.devices) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    from boxer_pipeline import BoxerPipeline

    labels = _labels_from_str(args.labels)
    pipeline = BoxerPipeline(
        ckpt_path=args.ckpt,
        detector=args.detector,
        labels=labels,
        thresh2d=args.thresh2d,
        thresh3d=args.thresh3d,
        track=args.track,
        skip_viz=args.skip_viz,
        output_dir=args.output_dir,
        stream_name="demo_sensor",
    )

    pipeline.on_result(make_result_handler(show_viz=args.show_viz))

    use_realsense = (args.input.lower() == "realsense") or (
        args.input == "0" and _try_realsense()
    )

    print(f"==> Detector : {args.detector}")
    print(f"==> Labels   : {labels[:8]}{'...' if len(labels) > 8 else ''}")
    print(f"==> Source   : {'RealSense' if use_realsense else f'OpenCV ({args.input})'}")
    print("Press Ctrl-C to stop.\n")

    with pipeline:
        if use_realsense:
            _run_realsense(args, pipeline)
        else:
            _run_opencv(args, pipeline)


if __name__ == "__main__":
    main()
