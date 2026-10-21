import argparse
import math
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path


ROOT = Path("/opt/rehiface")

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def run(args):
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA.")

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CUDA provider is unavailable.")

    from inference import FaceSwap, GenInput
    from options.hifi_test_options import HifiTestOptions

    original_arguments = sys.argv[:]

    try:
        sys.argv = [
            "inference.py",
            "--src_img_path",
            args.photo,
            "--video_path",
            args.video,
        ]
        options = HifiTestOptions().parse()
    finally:
        sys.argv = original_arguments

    capture = cv2.VideoCapture(args.video)

    if not capture.isOpened():
        raise ValueError("Cannot open target video.")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2 * 2
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2 * 2

    if (
        not math.isfinite(fps)
        or fps <= 0
        or min(width, height) < 2
    ):
        capture.release()
        raise ValueError("Invalid video dimensions or FPS.")

    input_queue = queue.Queue(maxsize=2)
    output_queue = queue.Queue(maxsize=2)
    source_features = []
    errors = queue.Queue()

    producer = GenInput(
        source_features,
        input_queue,
        output_queue,
        capture,
        args.photo,
    )

    engine = FaceSwap(
        source_features,
        input_queue,
        output_queue,
        model_name=options.model_name,
        align_method=options.align_method,
        use_gfpgan=args.restore,
        sr_weight=args.restore_weight,
        color_trans_mode=options.color_trans_mode,
    )

    def guarded(function):
        try:
            function()
        except BaseException:
            errors.put(traceback.format_exc())

    producer_thread = threading.Thread(
        target=guarded,
        args=(producer.run,),
        daemon=True,
    )

    engine_thread = threading.Thread(
        target=guarded,
        args=(engine.run,),
        daemon=True,
    )

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        args.video,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        args.output,
    ]

    process = None
    frame_count = 0
    started = time.monotonic()

    log_path = Path(args.output).with_suffix(".ffmpeg.log")

    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=log,
            )

            producer_thread.start()
            engine_thread.start()

            while True:
                if not errors.empty():
                    raise RuntimeError(errors.get())

                try:
                    frame = output_queue.get(timeout=1)
                except queue.Empty:
                    if not engine_thread.is_alive():
                        if not errors.empty():
                            raise RuntimeError(errors.get())
                        break

                    if time.monotonic() - started > args.timeout:
                        raise TimeoutError(
                            "ReHiFace-S processing exceeded its timeout."
                        )

                    continue

                if frame is None:
                    break

                if (
                    not isinstance(frame, np.ndarray)
                    or frame.ndim != 3
                    or frame.shape[2] != 3
                ):
                    raise RuntimeError(
                        "Unexpected output frame structure from engine."
                    )

                if frame.shape[0] < height or frame.shape[1] < width:
                    raise RuntimeError(
                        "Engine returned unexpected frame dimensions."
                    )

                pixels = np.ascontiguousarray(
                    frame[:height, :width],
                    dtype=np.uint8,
                )

                process.stdin.write(pixels.tobytes())
                frame_count += 1

                if frame_count % 25 == 0:
                    elapsed = time.monotonic() - started

                    print(
                        f"PROGRESS frames={frame_count} "
                        f"processing_fps={frame_count / max(elapsed, 0.001):.2f}",
                        flush=True,
                    )

            process.stdin.close()

            if process.wait(timeout=120) != 0:
                details = log_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )[-6000:]

                raise RuntimeError(f"FFmpeg failed:\n{details}")

        checker = cv2.VideoCapture(args.video)
        expected_frames = 0

        try:
            while True:
                success, _ = checker.read()

                if not success:
                    break

                expected_frames += 1
        finally:
            checker.release()

        if frame_count == 0 or frame_count != expected_frames:
            raise RuntimeError(
                "Incomplete output: "
                f"input_frames={expected_frames}, "
                f"output_frames={frame_count}."
            )

        output_path = Path(args.output)

        if (
            not output_path.is_file()
            or output_path.stat().st_size == 0
        ):
            raise RuntimeError("Encoded output is missing or empty.")

        elapsed = time.monotonic() - started

        print(
            f"DONE frames={frame_count} processing_seconds={elapsed:.2f}",
            flush=True,
        )

    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()

        capture.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--photo", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--restore",
        action="store_true",
    )

    parser.add_argument(
        "--restore-weight",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
    )

    arguments = parser.parse_args()

    try:
        run(arguments)
        exit_code = 0
    except BaseException:
        traceback.print_exc()
        exit_code = 1

    sys.stdout.flush()
    sys.stderr.flush()

    # Each job gets a disposable subprocess, including its engine threads.
    os._exit(exit_code)
