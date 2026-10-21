import base64
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen


ENDPOINT = os.environ["RUNPOD_ENDPOINT_ID"]
API_KEY = os.environ["RUNPOD_API_KEY"]

BASE_URL = "https://api.runpod.ai/v2/" + ENDPOINT


def api(path, body=None):
    payload = (
        None
        if body is None
        else json.dumps(body).encode("utf-8")
    )

    request = Request(
        BASE_URL + path,
        data=payload,
        headers={
            "Authorization": "Bearer " + API_KEY,
            "Content-Type": "application/json",
        },
    )

    with urlopen(request, timeout=120) as response:
        return json.load(response)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            'Usage: python client.py photo.jpg "Motion prompt"'
        )

    image = Path(sys.argv[1]).read_bytes()

    if len(image) > 4_000_000:
        raise ValueError("Use an image smaller than 4 MB.")

    prompt = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "The people smile and gently wave at the camera."
    )

    job = api(
        "/run",
        {
            "input": {
                "image_base64": base64.b64encode(
                    image
                ).decode("ascii"),
                "prompt": prompt,
                "frames": 49,
                "steps": 30,
                "seed": 42,
            }
        },
    )

    job_id = job["id"]

    print(
        "Submitted:",
        job_id,
        "\nClosing this program does not cancel the job.",
        flush=True,
    )

    deadline = time.monotonic() + 5400

    while time.monotonic() < deadline:
        time.sleep(5)

        job = api("/status/" + job_id)

        print(job["status"], flush=True)

        if job["status"] == "COMPLETED":
            video = base64.b64decode(
                job["output"]["video_base64"]
            )

            Path("result.mp4").write_bytes(video)

            print("Saved result.mp4")
            return

        if job["status"] in (
            "FAILED",
            "CANCELLED",
            "TIMED_OUT",
        ):
            raise RuntimeError(job)

    raise TimeoutError(
        "Polling stopped. Check Runpod job "
        + job_id
        + " before retrying."
    )


if __name__ == "__main__":
    main()
