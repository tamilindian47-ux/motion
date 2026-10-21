import base64
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import requests
import runpod


JOB_TIMEOUT = int(
    os.environ.get("JOB_TIMEOUT_SECONDS", "900")
)

MAX_INPUT_BYTES = int(
    os.environ.get("MAX_INPUT_BYTES", "268435456")
)

MAX_INLINE_OUTPUT_BYTES = int(
    os.environ.get("MAX_INLINE_OUTPUT_BYTES", "5000000")
)


def save_media(value, path):
    if not isinstance(value, str) or not value:
        raise ValueError("Both photo and video are required.")

    if value.startswith(("https://", "http://")):
        total_bytes = 0

        with requests.get(
            value,
            stream=True,
            timeout=(20, 120),
        ) as response:
            response.raise_for_status()

            with path.open("wb") as handle:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if not chunk:
                        continue

                    total_bytes += len(chunk)

                    if total_bytes > MAX_INPUT_BYTES:
                        raise ValueError(
                            "Input exceeds the worker's size limit."
                        )

                    handle.write(chunk)

    else:
        encoded = value

        if encoded.startswith("data:"):
            encoded = encoded.split(",", 1)[1]

        if len(encoded) > MAX_INPUT_BYTES * 4 // 3 + 4:
            raise ValueError("Base64 input is too large.")

        decoded = base64.b64decode(
            encoded,
            validate=True,
        )

        path.write_bytes(decoded)


def handler(event):
    try:
        job = event.get("input", {})

        restore = job.get("restore", False)
        restore_weight = float(
            job.get("restore_weight", 0.3)
        )

        if not isinstance(restore, bool):
            raise ValueError("restore must be true or false.")

        if not 0 <= restore_weight <= 1:
            raise ValueError(
                "restore_weight must be between 0 and 1."
            )

        with tempfile.TemporaryDirectory(
            prefix="rehiface-job-"
        ) as temporary_directory:
            directory = Path(temporary_directory)

            photo_path = directory / "source.jpg"
            video_path = directory / "target.mp4"
            output_path = directory / "result.mp4"
            log_path = directory / "engine.log"

            save_media(job.get("photo"), photo_path)
            save_media(job.get("video"), video_path)

            command = [
                sys.executable,
                "/app/runner.py",
                "--photo",
                str(photo_path),
                "--video",
                str(video_path),
                "--output",
                str(output_path),
                "--restore-weight",
                str(restore_weight),
                "--timeout",
                str(JOB_TIMEOUT),
            ]

            if restore:
                command.append("--restore")

            started = time.monotonic()

            with log_path.open("w+", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd="/opt/rehiface",
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

                try:
                    return_code = process.wait(
                        timeout=JOB_TIMEOUT + 180
                    )
                except subprocess.TimeoutExpired:
                    os.killpg(
                        process.pid,
                        signal.SIGKILL,
                    )
                    process.wait()

                    raise RuntimeError(
                        "Inference exceeded the worker timeout."
                    )

                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 16000))

                log_tail = log.read()

            if (
                return_code != 0
                or not output_path.is_file()
                or output_path.stat().st_size == 0
            ):
                raise RuntimeError(
                    f"ReHiFace-S failed, exit={return_code}.\n"
                    f"{log_tail}"
                )

            output_size = output_path.stat().st_size

            if output_size > MAX_INLINE_OUTPUT_BYTES:
                raise RuntimeError(
                    "Video processed but exceeds this trial's "
                    "inline-delivery limit. Use a shorter clip or "
                    "implement object-storage output. "
                    "No automatic quality reduction was performed."
                )

            result = base64.b64encode(
                output_path.read_bytes()
            ).decode("ascii")

            return {
                "status": "success",
                "media_type": "video",
                "base64": result,
                "metrics": {
                    "processing_seconds": round(
                        time.monotonic() - started,
                        2,
                    ),
                    "output_bytes": output_size,
                },
            }

    except Exception as exc:
        traceback.print_exc()

        return {
            "status": "failed",
            "error": str(exc),
        }


if __name__ == "__main__":
    runpod.serverless.start(
        {"handler": handler}
    )
