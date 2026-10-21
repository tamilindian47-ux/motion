import base64
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import runpod
from PIL import Image, ImageOps


Image.MAX_IMAGE_PIXELS = 25_000_000

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "/models/Wan2.2-TI2V-5B",
)


def handler(job):
    data = job.get("input", {})

    prompt = data.get("prompt", "")

    if (
        not isinstance(prompt, str)
        or not prompt.strip()
        or len(prompt) > 4000
    ):
        raise ValueError(
            "Provide a nonempty prompt of at most 4000 characters."
        )

    encoded = data.get("image_base64", "")

    if (
        not isinstance(encoded, str)
        or not encoded
        or len(encoded) > 6_000_000
    ):
        raise ValueError(
            "Provide a base64 image smaller than 6 million characters."
        )

    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]

    frames = int(data.get("frames", 49))
    steps = int(data.get("steps", 30))
    seed = int(data.get("seed", 42))

    if frames not in (49, 81, 121):
        raise ValueError("frames must be 49, 81 or 121.")

    if not 10 <= steps <= 50:
        raise ValueError("steps must be between 10 and 50.")

    if not 0 <= seed < 2**31:
        raise ValueError("seed must be between 0 and 2147483647.")

    with tempfile.TemporaryDirectory(prefix="wan-") as tmp:
        image_path = Path(tmp) / "input.png"
        video_path = Path(tmp) / "output.mp4"

        raw_image = base64.b64decode(
            encoded,
            validate=True,
        )

        with Image.open(io.BytesIO(raw_image)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((2048, 2048))
            image.save(image_path)

        runpod.serverless.progress_update(
            job,
            "Loading model and generating. Check worker logs.",
        )

        command = [
            sys.executable,
            "generate.py",
            "--task",
            "ti2v-5B",
            "--size",
            "1280*704",
            "--ckpt_dir",
            MODEL_DIR,
            "--offload_model",
            "True",
            "--convert_model_dtype",
            "--t5_cpu",
            "--image",
            str(image_path),
            "--prompt",
            prompt,
            "--frame_num",
            str(frames),
            "--sample_steps",
            str(steps),
            "--base_seed",
            str(seed),
            "--save_file",
            str(video_path),
        ]

        # No shell=True: the prompt is passed as one argument.
        result = subprocess.run(
            command,
            cwd="/app/Wan2.2",
            timeout=3300,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Wan generation failed. Inspect the worker logs."
            )

        if not video_path.is_file():
            raise RuntimeError(
                "No output MP4 was created. Inspect the worker logs."
            )

        video_bytes = video_path.read_bytes()

        if len(video_bytes) > 6_000_000:
            raise RuntimeError(
                "MP4 exceeds this starter's inline output limit. "
                "Use fewer frames or add object storage."
            )

        return {
            "video_base64": base64.b64encode(
                video_bytes
            ).decode("ascii"),
            "mime_type": "video/mp4",
            "frames": frames,
            "fps": 24,
            "seed": seed,
        }


if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
    })
