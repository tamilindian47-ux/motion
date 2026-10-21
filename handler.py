import os
import time
import json
import uuid
import base64
import requests
import subprocess
import runpod

COMFY_HOST = "127.0.0.1:8188"
INPUT_DIR = "/comfyui/input"
OUTPUT_DIR = "/comfyui/output"

# 1. Start headless ComfyUI locally inside the container
print("=== Booting headless ComfyUI engine ===", flush=True)
comfy_proc = subprocess.Popen(
    ["python3", "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch", "--gpu-only"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT,
)

# Wait until ComfyUI API is responsive
for _ in range(60):
    try:
        if requests.get(f"http://{COMFY_HOST}/system_stats", timeout=1).status_code == 200:
            print("=== ComfyUI engine is READY ===", flush=True)
            break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("ComfyUI failed to start in time.")

def save_payload(data_str, file_path):
    if data_str.startswith(("http://", "https://")):
        resp = requests.get(data_str, timeout=120)
        resp.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(resp.content)
    else:
        raw_b64 = data_str.split(",", 1)[1] if "," in data_str else data_str
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(raw_b64))

def handler(event):
    job_input = event.get("input", {})
    photo_payload = job_input.get("photo")
    video_payload = job_input.get("video")

    if not photo_payload or not video_payload:
        return {"status": "failed", "error": "Both 'photo' and 'video' must be provided."}

    session_id = str(uuid.uuid4())[:8]
    ref_filename = f"ref_{session_id}.jpg"
    vid_filename = f"vid_{session_id}.mp4"

    ref_path = os.path.join(INPUT_DIR, ref_filename)
    vid_path = os.path.join(INPUT_DIR, vid_filename)

    save_payload(photo_payload, ref_path)
    save_payload(video_payload, vid_path)

    # Load base workflow
    with open("/comfyui/workflow_api.json", "r") as f:
        prompt_graph = json.load(f)

    # Dynamically map incoming input files into graph nodes
    # Node 1: Video Loader | Node 2: Reference Image Loader
    prompt_graph["1"]["inputs"]["video"] = vid_filename
    prompt_graph["2"]["inputs"]["image"] = ref_filename
    prompt_graph["10"]["inputs"]["filename_prefix"] = f"out_{session_id}"

    # Queue the job in ComfyUI
    req = requests.post(f"http://{COMFY_HOST}/prompt", json={"prompt": prompt_graph})
    prompt_id = req.json().get("prompt_id")

    # Poll history until finished
    completed = False
    output_filename = None
    for _ in range(600):  # 10 min timeout
        time.sleep(1)
        history = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}").json()
        if prompt_id in history:
            completed = True
            # Extract video filename from VideoCombine node output
            outputs = history[prompt_id].get("outputs", {})
            for node_id, out in outputs.items():
                if "gifs" in out:
                    output_filename = out["gifs"][0]["filename"]
                elif "videos" in out:
                    output_filename = out["videos"][0]["filename"]
            break

    if not completed or not output_filename:
        return {"status": "failed", "error": "Execution timed out or video failed to render."}

    result_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(result_path, "rb") as f:
        encoded_video = base64.b64encode(f.read()).decode("utf-8")

    # Clean up working files to keep disk free
    for p in [ref_path, vid_path, result_path]:
        if os.path.exists(p):
            os.remove(p)

    return {
        "status": "success",
        "video_base64": encoded_video
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
