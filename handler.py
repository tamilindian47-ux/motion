import base64
import binascii
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import requests
import runpod
import torch
from huggingface_hub import snapshot_download
from PIL import Image


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
LOG = logging.getLogger("wan-animate-worker")

WAN_ROOT = Path(
    os.environ.get("WAN_ROOT", "/app/Wan2.2")
).resolve()

MODEL_REPO = "Wan-AI/Wan2.2-Animate-14B"

MODEL_DIR = Path(
    os.environ.get("WAN_MODEL_DIR", "/tmp/wan-animate-model")
).resolve()

MODEL_REVISION = os.environ.get("WAN_MODEL_REVISION", "main")

PREPROCESS_SCRIPT = (
    WAN_ROOT
    / "wan"
    / "modules"
    / "animate"
    / "preprocess"
    / "preprocess_data.py"
)

GENERATE_SCRIPT = WAN_ROOT / "generate.py"

MAX_IMAGE_BYTES = int(
    os.environ.get("MAX_IMAGE_BYTES", str(8 * 1024 * 1024))
)

MAX_VIDEO_BYTES = int(
    os.environ.get("MAX_VIDEO_BYTES", str(64 * 1024 * 1024))
)

# Base64 adds roughly one third to the binary output size.
# Set below the response limit of your actual RunPod endpoint.
MAX_OUTPUT_BYTES = int(
    os.environ.get("MAX_OUTPUT_BYTES", str(5 * 1024 * 1024))
)

MAX_INPUT_SECONDS = float(
    os.environ.get("MAX_INPUT_SECONDS", "5")
)

STAGE_TIMEOUT_SECONDS = int(
    os.environ.get("STAGE_TIMEOUT_SECONDS", "1800")
)

MIN_FREE_GB = float(
    os.environ.get("MIN_FREE_GB", "10")
)

JOB_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()

MODEL_READY = False


def log_progress(job, message):
    LOG.info(message)

    try:
        if job.get("id"):
            runpod.serverless.progress_update(
                job,
                {"stage": message},
            )
    except Exception:
        LOG.debug("Progress update unavailable", exc_info=True)


def run_command(command, stage, timeout=STAGE_TIMEOUT_SECONDS):
    """
    Execute a fixed argument list without shell interpolation.
    Child stdout/stderr is streamed to worker logs.
    """
    command = [str(item) for item in command]

    LOG.info(
        "%s command: %s",
        stage,
        json.dumps(command),
    )

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    previous_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(WAN_ROOT)
        + (os.pathsep + previous_pythonpath if previous_pythonpath else "")
    )

    tail = deque(maxlen=60)
    timed_out = threading.Event()

    process = subprocess.Popen(
        command,
        cwd=str(WAN_ROOT),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )

    def terminate_on_timeout():
        timed_out.set()
        try:
            process.kill()
        except ProcessLookupError:
            pass

    timer = threading.Timer(timeout, terminate_on_timeout)
    timer.daemon = True
    timer.start()

    try:
        for line in process.stdout:
            line = line.rstrip()
            tail.append(line)
            LOG.info("[%s] %s", stage, line)

        return_code = process.wait()
    finally:
        timer.cancel()
        if process.stdout:
            process.stdout.close()

    if timed_out.is_set():
        raise RuntimeError(
            f"{stage} exceeded {timeout} seconds.\n"
            + "\n".join(tail)
        )

    if return_code != 0:
        raise RuntimeError(
            f"{stage} failed with exit code {return_code}.\n"
            + "\n".join(tail)
        )


def verify_environment():
    missing = [
        str(path)
        for path in (PREPROCESS_SCRIPT, GENERATE_SCRIPT)
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            "This container does not contain the required official "
            "Wan2.2 source files:\n"
            + "\n".join(missing)
            + "\nChanging handler.py alone cannot install the animation "
            "backend. Install the Wan2.2 source and its dependencies "
            "during the Docker build."
        )

    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(
                f"Required executable is missing: {executable}"
            )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    LOG.info("Actual torch runtime: %s", torch.__version__)
    LOG.info("Bundled CUDA: %s", torch.version.cuda)
    LOG.info("GPU: %s", torch.cuda.get_device_name(0))
    LOG.info(
        "GPU capability: %s",
        torch.cuda.get_device_capability(0),
    )

    try:
        tensor = torch.ones((32, 32), device="cuda")
        result = tensor @ tensor
        assert result[0, 0].item() == 32.0
        torch.cuda.synchronize()
        del tensor, result
        torch.cuda.empty_cache()
    except Exception as exc:
        raise RuntimeError(
            "The installed PyTorch build cannot execute CUDA kernels "
            "on the assigned GPU. Do not spoof torch.__version__ or "
            "disable the health check."
        ) from exc

    # Test the actual repository entry points before downloading weights.
    run_command(
        [sys.executable, PREPROCESS_SCRIPT, "--help"],
        "preprocessor dependency check",
        timeout=180,
    )

    run_command(
        [sys.executable, GENERATE_SCRIPT, "--help"],
        "generator dependency check",
        timeout=180,
    )


def ensure_model(job):
    global MODEL_READY

    with MODEL_LOCK:
        if MODEL_READY:
            return

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        free_gb = (
            shutil.disk_usage(MODEL_DIR).free / (1024 ** 3)
        )

        if free_gb < MIN_FREE_GB:
            raise RuntimeError(
                f"Only {free_gb:.1f} GiB free at {MODEL_DIR}. "
                "Insufficient working-space reserve."
            )

        log_progress(job, "Downloading/checking Wan-Animate checkpoint")

        # Invoke this even if the directory already exists:
        # an existing directory does not prove a download completed.
        snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            local_dir=str(MODEL_DIR),
            max_workers=4,
        )

        process_checkpoint = MODEL_DIR / "process_checkpoint"

        if not process_checkpoint.is_dir():
            raise RuntimeError(
                "The downloaded checkpoint is missing "
                f"{process_checkpoint}. "
                "The required model is Wan2.2-Animate-14B, "
                "not a Wan2.1 I2V Diffusers repository."
            )

        MODEL_READY = True


def write_input(raw, destination, max_bytes):
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Input must be a non-empty URL or base64 string.")

    raw = raw.strip()
    destination = Path(destination)

    if raw.startswith(("http://", "https://")):
        # This direct URL mode is intended for a private,
        # owner-operated endpoint. Do not expose it publicly.
        with requests.get(
            raw,
            stream=True,
            timeout=(15, 120),
        ) as response:
            response.raise_for_status()

            written = 0
            with destination.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue

                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(
                            f"Input exceeds {max_bytes} bytes."
                        )

                    output.write(chunk)

    else:
        if raw.startswith("data:"):
            if "," not in raw:
                raise ValueError("Malformed data URL.")
            raw = raw.split(",", 1)[1]

        if len(raw) > 4 * ((max_bytes + 2) // 3) + 16:
            raise ValueError("Encoded input exceeds the size limit.")

        try:
            data = base64.b64decode(raw, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid base64 input.") from exc

        if len(data) > max_bytes:
            raise ValueError("Decoded input exceeds the size limit.")

        destination.write_bytes(data)

    if not destination.is_file() or destination.stat().st_size == 0:
        raise ValueError("Input file is empty.")


def validate_image(path):
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        rgb.save(path, format="PNG")


def inspect_video(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height:format=duration",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

    metadata = json.loads(result.stdout)
    streams = metadata.get("streams", [])

    if not streams:
        raise ValueError("Driving file contains no video stream.")

    duration = float(metadata.get("format", {}).get("duration", 0))

    if duration <= 0:
        raise ValueError("Cannot determine driving-video duration.")

    if duration > MAX_INPUT_SECONDS + 0.1:
        raise ValueError(
            f"Driving clip is {duration:.2f}s. "
            f"This deployment currently permits {MAX_INPUT_SECONDS:.1f}s. "
            "Trim the clip before uploading."
        )

    return {
        "duration": duration,
        "width": int(streams[0]["width"]),
        "height": int(streams[0]["height"]),
    }


def integer_parameter(data, name, default, lower, upper):
    value = int(data.get(name, default))
    if not lower <= value <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}."
        )
    return value


def handler(job):
    started = time.monotonic()

    # One job at a time per GPU worker.
    with JOB_LOCK:
        data = job.get("input") or {}

        mode = str(data.get("mode", "animate")).lower()
        if mode not in {"animate", "replace"}:
            raise ValueError("mode must be 'animate' or 'replace'.")

        character_image = data.get("character_image")
        driving_video = data.get("driving_video")

        if not character_image or not driving_video:
            raise ValueError(
                "Both character_image and driving_video are required."
            )

        width = integer_parameter(data, "width", 1280, 256, 1280)
        height = integer_parameter(data, "height", 720, 256, 1280)
        steps = integer_parameter(
            data, "num_inference_steps", 20, 1, 50
        )
        seed = integer_parameter(
            data, "seed", 42, -1, 2147483647
        )

        log_progress(job, "Checking animation backend and GPU")
        verify_environment()

        with tempfile.TemporaryDirectory(
            prefix="wan-animate-job-"
        ) as working_directory:
            work = Path(working_directory)
            image_path = work / "character.png"
            video_path = work / "driving.mp4"
            process_dir = work / "processed"
            output_path = work / "output.mp4"

            log_progress(job, "Validating reference image and driving video")

            write_input(
                character_image,
                image_path,
                MAX_IMAGE_BYTES,
            )
            validate_image(image_path)

            write_input(
                driving_video,
                video_path,
                MAX_VIDEO_BYTES,
            )
            source_metadata = inspect_video(video_path)

            ensure_model(job)

            preprocessing_command = [
                sys.executable,
                PREPROCESS_SCRIPT,
                "--ckpt_path",
                MODEL_DIR / "process_checkpoint",
                "--video_path",
                video_path,
                "--refer_path",
                image_path,
                "--save_path",
                process_dir,
                "--resolution_area",
                width,
                height,
            ]

            if mode == "replace":
                preprocessing_command.extend([
                    "--iterations", "3",
                    "--k", "7",
                    "--w_len", "1",
                    "--h_len", "1",
                    "--replace_flag",
                ])
            else:
                preprocessing_command.append("--retarget_flag")

            log_progress(job, f"Preprocessing driving video: {mode}")
            run_command(
                preprocessing_command,
                "Wan-Animate preprocessing",
            )

            expected = [
                "src_ref.png",
                "src_pose.mp4",
                "src_face.mp4",
            ]
            if mode == "replace":
                expected.extend(["src_bg.mp4", "src_mask.mp4"])

            missing = [
                name for name in expected
                if not (process_dir / name).is_file()
            ]
            if missing:
                raise RuntimeError(
                    "Preprocessing did not produce required files: "
                    + ", ".join(missing)
                )

            generation_command = [
                sys.executable,
                GENERATE_SCRIPT,
                "--task", "animate-14B",
                "--ckpt_dir", MODEL_DIR,
                "--src_root_path", process_dir,
                "--refert_num", "1",
                "--offload_model", "True",
                "--convert_model_dtype",
                "--t5_cpu",
                "--sample_steps", steps,
                "--base_seed", seed,
                "--save_file", output_path,
            ]

            if mode == "replace":
                generation_command.extend([
                    "--replace_flag",
                    "--use_relighting_lora",
                ])

            log_progress(job, f"Generating Wan-Animate video: {mode}")
            run_command(
                generation_command,
                "Wan-Animate generation",
            )

            if (
                not output_path.is_file()
                or output_path.stat().st_size == 0
            ):
                raise RuntimeError("Generation produced no output video.")

            output_size = output_path.stat().st_size

            if output_size > MAX_OUTPUT_BYTES:
                raise RuntimeError(
                    f"Generated video is {output_size} bytes, exceeding "
                    f"the configured inline return limit "
                    f"({MAX_OUTPUT_BYTES} bytes). "
                    "Use an object-storage return path for larger outputs; "
                    "do not send oversized base64 responses."
                )

            encoded = base64.b64encode(
                output_path.read_bytes()
            ).decode("ascii")

            return {
                "status": "success",
                "mode": mode,
                "model": MODEL_REPO,
                "video_base64": encoded,
                "mime_type": "video/mp4",
                "output_bytes": output_size,
                "source_video": source_metadata,
                "render_time_sec": round(
                    time.monotonic() - started, 2
                ),
                "notes": [
                    "Output audio is not restored by this handler.",
                    "Prompt and negative_prompt inputs are not used "
                    "by the official animation generation branch.",
                ],
            }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
