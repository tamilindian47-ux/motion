import os
import sys
import time
import json
import base64
import shutil
import urllib.request
import urllib.parse
import subprocess
from pathlib import Path
import runpod
from huggingface_hub import hf_hub_download

COMFY_ROOT = Path("/app/ComfyUI")
INPUT_DIR = COMFY_ROOT / "input"
OUTPUT_DIR = COMFY_ROOT / "output"
COMFY_PORT = 8188
SERVER_URL = f"http://127.0.0.1:{COMFY_PORT}"

COMFY_PROC = None
READY = False

def download_models(job):
    """Fetch weights inside the worker job, reporting progress to RunPod."""
    models_to_fetch = [
        ("realrebelai/SCAIL-2_GGUF", "scail-2-q4_k_m.gguf", COMFY_ROOT / "models" / "unet"),
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", COMFY_ROOT / "models" / "clip"),
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors", COMFY_ROOT / "models" / "clip_vision"),
        ("Kijai/WanVideo_comfy", "Wan2_1_VAE_bf16.safetensors", COMFY_ROOT / "models" / "vae")
    ]

    for repo, filename, dest_dir in models_to_fetch:
        dest_dir.mkdir(parents=True, exist_ok=True)
        local_name = Path(filename).name
        target_path = dest_dir / local_name

        if not target_path.exists() or target_path.stat().st_size == 0:
            msg = f"Downloading {local_name} from {repo} to ephemeral disk..."
            print(f"[Loader] {msg}", flush=True)
            try:
                runpod.serverless.progress_update(job, {"stage": msg})
            except Exception:
                pass

            downloaded = hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=str(dest_dir),
                local_dir_use_symlinks=False
            )
            if Path(downloaded) != target_path:
                shutil.move(downloaded, target_path)
            print(f"[Loader] {local_name} ready.", flush=True)

def ensure_comfy_ready(job):
    """Ensure models exist and ComfyUI background process is active."""
    global COMFY_PROC, READY
    if READY:
        return

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    download_models(job)

    if COMFY_PROC is None:
        print("[Server] Starting headless ComfyUI on port 8188...", flush=True)
        cmd = [
            sys.executable,
            str(COMFY_ROOT / "main.py"),
            "--listen", "127.0.0.1",
            "--port", str(COMFY_PORT),
            "--disable-auto-launch",
            "--highvram"
        ]
        COMFY_PROC = subprocess.Popen(cmd, cwd=str(COMFY_ROOT))

        # Wait for local HTTP API readiness
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"{SERVER_URL}/system_stats", timeout=2) as resp:
                    if resp.status == 200:
                        print("[Server] ComfyUI local API listening and ready.", flush=True)
                        READY = True
                        return
            except Exception:
                time.sleep(1)

        raise RuntimeError("ComfyUI server failed to respond within 60s.")

def queue_prompt(prompt_workflow):
    payload = json.dumps({"prompt": prompt_workflow}).encode("utf-8")
    req = urllib.request.Request(f"{SERVER_URL}/prompt", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))

def wait_for_execution(prompt_id, timeout=900):
    start_time = time.time()
    while time.time() - start_time < timeout:
        time.sleep(2)
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/history/{prompt_id}") as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if prompt_id in data:
                    return data[prompt_id]
        except Exception:
            continue
    raise TimeoutError(f"Generation timed out after {timeout} seconds.")

def write_b64_file(raw_data, dest_path):
    if "," in raw_data:
        raw_data = raw_data.split(",", 1)[1]
    decoded = base64.b64decode(raw_data)
    with open(dest_path, "wb") as f:
        f.write(decoded)

def handler(job):
    job_input = job.get("input", {})
    char_img_b64 = job_input.get("character_image")
    driving_vid_b64 = job_input.get("driving_video")

    if not char_img_b64 or not driving_vid_b64:
        return {"error": "Missing 'character_image' or 'driving_video' base64 payload."}

    # 1. Warm boot models and start ComfyUI server if not already alive
    ensure_comfy_ready(job)

    # 2. Clean previous outputs
    for p in OUTPUT_DIR.glob("*.mp4"):
        try:
            p.unlink()
        except Exception:
            pass

    # 3. Write input assets
    char_img_path = INPUT_DIR / "character.png"
    driving_vid_path = INPUT_DIR / "driving.mp4"
    write_b64_file(char_img_b64, char_img_path)
    write_b64_file(driving_vid_b64, driving_vid_path)

    # 4. Inject runtime parameters into workflow graph
    with open("/app/workflow_api.json", "r") as f:
        workflow = json.load(f)

    steps = int(job_input.get("num_inference_steps", 20))
    seed = int(job_input.get("seed", 42))
    w = int(job_input.get("width", 512))
    h = int(job_input.get("height", 512))

    workflow["11"]["inputs"]["steps"] = steps
    workflow["11"]["inputs"]["seed"] = seed
    workflow["10"]["inputs"]["width"] = w
    workflow["10"]["inputs"]["height"] = h
    workflow["6"]["inputs"]["custom_width"] = w
    workflow["6"]["inputs"]["custom_height"] = h

    print(f"[Job] Submitting prompt to ComfyUI (Steps: {steps}, Seed: {seed})...", flush=True)
    res = queue_prompt(workflow)
    prompt_id = res.get("prompt_id")
    if not prompt_id:
        return {"error": "Failed to queue workflow", "details": res}

    history = wait_for_execution(prompt_id, timeout=900)

    # 5. Extract output video
    mp4_files = list(OUTPUT_DIR.glob("*.mp4"))
    if not mp4_files:
        return {"error": "Execution completed but no MP4 was written.", "history": history}

    latest_mp4 = max(mp4_files, key=os.path.getmtime)
    with open(latest_mp4, "rb") as vf:
        out_b64 = base64.b64encode(vf.read()).decode("utf-8")

    return {
        "status": "success",
        "video_base64": out_b64,
        "filename": latest_mp4.name
    }

# Start the RunPod worker loop immediately so fitness check passes
runpod.serverless.start({"handler": handler})
