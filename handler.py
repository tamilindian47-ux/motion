import base64
import gc
import io
import logging
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

import runpod
import torch
from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
)
from diffusers.utils import export_to_video
from PIL import Image, ImageOps


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("wan-turbo")

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/wan-turbo")

Image.MAX_IMAGE_PIXELS = 25_000_000

PIPELINE = None
LOCK = threading.Lock()

MAX_INPUT_BYTES = 4_000_000
MAX_VIDEO_BYTES = 6_000_000


def progress(job, message):
    log.info(message)
    runpod.serverless.progress_update(job, message)


def load_pipeline():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected.")

    log.info("GPU: %s", torch.cuda.get_device_name(0))

    # FP32 VAE prioritizes decoding stability over minimum memory.
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_DIR,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )

    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_DIR,
        vae=vae,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=5.0,
    )

    pipe.vae.enable_tiling()

    # Do not also call pipe.to("cuda").
    # This retains the Python pipeline while moving components
    # between CPU and GPU as needed.
    pipe.enable_model_cpu_offload(gpu_id=0)

    return pipe


def decode_image(encoded):
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("image_base64 is required.")

    if len(encoded) > 5_400_000:
        raise ValueError("Image is too large. Use an image under 4 MB.")

    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]

    raw = base64.b64decode(encoded, validate=True)

    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("Image must be under 4 MB.")

    with Image.open(io.BytesIO(raw)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()

    if min(image.size) < 64:
        raise ValueError("Image dimensions are too small.")

    ratio = image.width / image.height

    if not 0.4 <= ratio <= 2.5:
        raise ValueError(
            "Use a normal portrait or landscape image, "
            "not an extreme panorama."
        )

    return image


def prepare_image(image, pipe):
    max_area = 1280 * 704

    multiple = int(
        pipe.vae_scale_factor_spatial
        * pipe.transformer.config.patch_size[1]
    )

    ratio = image.width / image.height

    width = (
        round(math.sqrt(max_area * ratio))
        // multiple
        * multiple
    )

    height = (
        round(math.sqrt(max_area / ratio))
        // multiple
        * multiple
    )

    width = max(multiple, width)
    height = max(multiple, height)

    image = image.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    return image, width, height


def fit_inline_output(source_path, folder):
    """
    Prefer the original encoding. If it exceeds the inline budget,
    retry H.264 compression without changing frame rate or dimensions.
    """
    if source_path.stat().st_size <= MAX_VIDEO_BYTES:
        return source_path.read_bytes()

    for crf in (24, 28, 32, 36):
        target = folder / f"compressed-{crf}.mp4"

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(target),
            ],
            check=True,
            timeout=180,
        )

        if target.stat().st_size <= MAX_VIDEO_BYTES:
            return target.read_bytes()

    raise RuntimeError(
        "Video exceeds the inline output budget even after compression. "
        "Use a shorter clip or implement object-storage output."
    )


def handler(job):
    global PIPELINE

    data = job.get("input", {})

    prompt = data.get("prompt", "")

    if (
        not isinstance(prompt, str)
        or not prompt.strip()
        or len(prompt) > 4000
    ):
        raise ValueError(
            "Provide a prompt between 1 and 4000 characters."
        )

    frames = int(data.get("frames", 49))
    seed = int(data.get("seed", 42))

    if frames not in (49, 81, 121):
        raise ValueError("frames must be 49, 81 or 121.")

    if not 0 <= seed < 2**31:
        raise ValueError("seed must be between 0 and 2147483647.")

    image = decode_image(data.get("image_base64"))

    # One GPU generation at a time.
    with LOCK:
        job_started = time.perf_counter()
        load_seconds = 0.0
        cold_model_load = PIPELINE is None

        try:
            if PIPELINE is None:
                progress(job, "Loading Turbo model for this worker.")

                load_started = time.perf_counter()
                candidate = load_pipeline()

                # Assign only after initialization succeeds.
                PIPELINE = candidate
                load_seconds = time.perf_counter() - load_started

            pipe = PIPELINE

            image, width, height = prepare_image(image, pipe)

            torch.cuda.reset_peak_memory_stats()

            progress(
                job,
                f"Generating {frames} frames at {width}x{height}, "
                "using four Turbo steps.",
            )

            generation_started = time.perf_counter()

            generator = torch.Generator(
                device="cuda"
            ).manual_seed(seed)

            with torch.inference_mode():
                result = pipe(
                    image=image,
                    prompt=prompt.strip(),
                    width=width,
                    height=height,
                    num_frames=frames,
                    num_inference_steps=4,
                    guidance_scale=1.0,
                    generator=generator,
                )

            video_frames = result.frames[0]

            torch.cuda.synchronize()

            generation_seconds = (
                time.perf_counter() - generation_started
            )

            peak_vram_gb = (
                torch.cuda.max_memory_allocated() / (1024**3)
            )

            progress(job, "Encoding MP4.")

            encode_started = time.perf_counter()

            with tempfile.TemporaryDirectory(
                prefix="wan-turbo-"
            ) as temp:
                folder = Path(temp)
                video_path = folder / "video.mp4"

                export_to_video(
                    video_frames,
                    str(video_path),
                    fps=24,
                )

                video_bytes = fit_inline_output(
                    video_path,
                    folder,
                )

            encode_seconds = (
                time.perf_counter() - encode_started
            )

            del result, video_frames

            response = {
                "video_base64": base64.b64encode(
                    video_bytes
                ).decode("ascii"),
                "mime_type": "video/mp4",
                "width": width,
                "height": height,
                "frames": frames,
                "fps": 24,
                "seed": seed,
                "steps": 4,
                "cold_model_load": cold_model_load,
                "timing": {
                    "model_load_seconds": round(load_seconds, 2),
                    "generation_seconds": round(
                        generation_seconds, 2
                    ),
                    "encode_seconds": round(encode_seconds, 2),
                    "handler_seconds": round(
                        time.perf_counter() - job_started, 2
                    ),
                },
                "peak_allocated_vram_gb": round(
                    peak_vram_gb, 2
                ),
            }

            log.info(
                "Completed: %s",
                response["timing"],
            )

            return response

        except torch.cuda.OutOfMemoryError:
            # Offload hooks can be left in an intermediate state
            # after failure. Force a fresh model on the next request.
            PIPELINE = None

            log.exception("CUDA out of memory")

            raise RuntimeError(
                "CUDA out of memory. This configuration has not "
                "been verified to fit your GPU. Do not repeatedly "
                "retry the same failed configuration."
            )

        finally:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
    })
