import os

# Set these before importing Hugging Face libraries.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

BUILD_VERSION = "ltx-no-slicing-v2"
print(f"HANDLER BUILD: {BUILD_VERSION}", flush=True)

import base64
import binascii
import io
import shutil
import tempfile
import threading
import traceback

import runpod
import torch
from PIL import Image, ImageOps
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download


MODEL_ID = "Lightricks/LTX-Video"

# No network volume required.
# Use /runpod-volume only when one is available; otherwise use local storage.
STORAGE_ROOT = (
    "/runpod-volume"
    if os.path.isdir("/runpod-volume")
    else "/app/models_cache"
)

MODEL_DIR = os.path.join(STORAGE_ROOT, "ltx_video_diffusers")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

PIPELINE = None
PIPELINE_LOCK = threading.Lock()

DOWNLOAD_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "transformer/*",
    "vae/*",
]


def log(message):
    print(f"[{BUILD_VERSION}] {message}", flush=True)


def get_or_load_pipeline():
    global PIPELINE

    if PIPELINE is not None:
        return PIPELINE

    os.makedirs(MODEL_DIR, exist_ok=True)

    log(f"Model directory: {MODEL_DIR}")
    log(f"Device: {DEVICE}; dtype: {DTYPE}")
    log("Checking/downloading model files...")

    # Let the downloader check individual files rather than assuming
    # model_index.json means that every weight file is present.
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        allow_patterns=DOWNLOAD_PATTERNS,
        max_workers=4,
    )

    log("Loading LTXImageToVideoPipeline...")

    # Assign the global only after initialization succeeds.
    pipeline = LTXImageToVideoPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=DTYPE,
        local_files_only=True,
    )

    pipeline = pipeline.to(DEVICE)

    # Intentionally no enable_vae_slicing() call.
    # Keep initialization minimal while testing the first successful output.

    PIPELINE = pipeline
    log("Pipeline loaded.")

    return PIPELINE


def decode_image(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Input 'image' must be a non-empty base64 string.")

    value = value.strip()

    if value.startswith("data:"):
        if "," not in value:
            raise ValueError("Invalid image data URL.")

        header, value = value.split(",", 1)

        if ";base64" not in header.lower():
            raise ValueError("The image data URL must use base64 encoding.")

    # Allow line breaks or whitespace in a base64 payload.
    value = "".join(value.split())

    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Could not decode the base64 image.") from exc

    try:
        with Image.open(io.BytesIO(raw)) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise ValueError("Decoded input is not a readable image.") from exc


def read_int(data, name, default, minimum, maximum):
    value = data.get(name, default)

    if isinstance(value, bool):
        raise ValueError(f"'{name}' must be an integer.")

    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"'{name}' must be an integer.") from exc

    if isinstance(value, float) and value != parsed:
        raise ValueError(f"'{name}' must be an integer.")

    if not minimum <= parsed <= maximum:
        raise ValueError(
            f"'{name}' must be between {minimum} and {maximum}."
        )

    return parsed


def read_float(data, name, default, minimum, maximum):
    try:
        value = float(data.get(name, default))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"'{name}' must be a number.") from exc

    # This also rejects NaN and infinities.
    if not minimum <= value <= maximum:
        raise ValueError(
            f"'{name}' must be between {minimum} and {maximum}."
        )

    return value


def handler(event):
    work_dir = None

    try:
        job_input = event.get("input", {})

        if not isinstance(job_input, dict):
            raise ValueError("'input' must be a JSON object.")

        current_image = decode_image(job_input.get("image"))

        prompt = job_input.get(
            "prompt",
            "High quality cinematic motion, smooth camera movement",
        )

        negative_prompt = job_input.get(
            "negative_prompt",
            "worst quality, blurry, jittery, distorted, low resolution",
        )

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("'prompt' must be a non-empty string.")

        if not isinstance(negative_prompt, str):
            raise ValueError("'negative_prompt' must be a string.")

        width = read_int(job_input, "width", 768, 256, 1024)
        height = read_int(job_input, "height", 512, 256, 1024)

        if width % 32 or height % 32:
            raise ValueError(
                "'width' and 'height' must be multiples of 32."
            )

        duration_seconds = read_int(
            job_input, "duration_seconds", 5, 5, 15
        )

        if duration_seconds not in (5, 10, 15):
            raise ValueError(
                "'duration_seconds' must be 5, 10, or 15."
            )

        fps = read_int(job_input, "fps", 24, 24, 24)

        num_inference_steps = read_int(
            job_input, "num_inference_steps", 30, 1, 100
        )

        guidance_scale = read_float(
            job_input, "guidance_scale", 3.0, 1.0, 10.0
        )

        seed = read_int(
            job_input, "seed", 42, 0, (2**63) - 1
        )

        num_segments = duration_seconds // 5
        frames_per_segment = 121
        target_frame_count = duration_seconds * fps

        # Fit the reference image without stretching its proportions.
        current_image = ImageOps.fit(
            current_image,
            (width, height),
            method=Image.Resampling.LANCZOS,
        )

        work_dir = tempfile.mkdtemp(prefix="ltx_")
        output_path = os.path.join(work_dir, "output.mp4")

        log(
            f"Starting job: {duration_seconds}s, {width}x{height}, "
            f"{num_segments} segment(s), {num_inference_steps} steps"
        )

        # Serialize access to the shared pipeline in this process.
        with PIPELINE_LOCK:
            pipe = get_or_load_pipeline()
            stitched_frames = []

            with torch.inference_mode():
                for segment_idx in range(num_segments):
                    log(
                        f"Generating segment "
                        f"{segment_idx + 1}/{num_segments}..."
                    )

                    generator = torch.Generator(
                        device=DEVICE
                    ).manual_seed(seed + segment_idx * 100)

                    result = pipe(
                        image=current_image,
                        prompt=prompt.strip(),
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_frames=frames_per_segment,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        output_type="pil",
                    )

                    segment_frames = list(result.frames[0])

                    if not segment_frames:
                        raise RuntimeError(
                            f"Segment {segment_idx + 1} returned no frames."
                        )

                    if not all(
                        isinstance(frame, Image.Image)
                        for frame in segment_frames
                    ):
                        raise RuntimeError(
                            "Expected PIL images from output_type='pil'."
                        )

                    # Use the final frame as the next segment's reference.
                    current_image = segment_frames[-1].copy()

                    if segment_idx == 0:
                        stitched_frames.extend(segment_frames)
                    else:
                        # Omit the repeated starting frame.
                        stitched_frames.extend(segment_frames[1:])

                    log(
                        f"Segment {segment_idx + 1} complete; "
                        f"{len(stitched_frames)} accumulated frames."
                    )

                    del result

            if len(stitched_frames) < target_frame_count:
                raise RuntimeError(
                    f"Expected at least {target_frame_count} frames, "
                    f"received {len(stitched_frames)}."
                )

            # Export exactly the requested duration at 24 fps.
            stitched_frames = stitched_frames[:target_frame_count]

            log(
                f"Exporting {len(stitched_frames)} frames "
                f"at {fps} fps..."
            )

            export_to_video(
                stitched_frames,
                output_path,
                fps=fps,
            )

            del stitched_frames

        if (
            not os.path.isfile(output_path)
            or os.path.getsize(output_path) == 0
        ):
            raise RuntimeError("Video export did not produce a valid file.")

        with open(output_path, "rb") as video_file:
            video_base64 = base64.b64encode(
                video_file.read()
            ).decode("utf-8")

        log("Job complete.")

        return {
            "status": "success",
            "build_version": BUILD_VERSION,
            "video_base64": video_base64,
            "filename": f"ltx_video_{duration_seconds}s.mp4",
            "total_frames": target_frame_count,
            "duration_seconds": duration_seconds,
            "fps": fps,
            "seed": seed,
        }

    except Exception:
        log("Job failed:")
        traceback.print_exc()

        # Propagate the exception so the job is marked failed,
        # rather than returning a successful result containing an error.
        raise

    finally:
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    log("Starting RunPod handler.")
    runpod.serverless.start({"handler": handler})
