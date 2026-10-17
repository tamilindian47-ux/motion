import os
import io
import time
import base64
import tempfile
import requests
import cv2
import torch
import numpy as np
import runpod
from PIL import Image
from transformers import CLIPVisionModel
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download

MODEL_REPO = os.getenv("MODEL_REPO", "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers")
LOCAL_DIR = "/tmp/wan21_model"
PIPELINE = None

def get_pipeline():
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    if not os.path.exists(LOCAL_DIR) or len(os.listdir(LOCAL_DIR)) == 0:
        print(f"Downloading model {MODEL_REPO} to ephemeral storage...")
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True
        )

    print("Loading pipeline components into memory...")
    vae = AutoencoderKLWan.from_pretrained(
        LOCAL_DIR, 
        subfolder="vae", 
        torch_dtype=torch.float32
    )
    
    image_encoder = CLIPVisionModel.from_pretrained(
        LOCAL_DIR, 
        subfolder="image_encoder", 
        torch_dtype=torch.float32
    )

    pipe = WanImageToVideoPipeline.from_pretrained(
        LOCAL_DIR,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=torch.bfloat16
    )

    pipe.enable_model_cpu_offload()
    PIPELINE = pipe
    return PIPELINE

def safe_dimensions(width: int, height: int):
    w = (int(width) // 16) * 16
    h = (int(height) // 16) * 16
    return max(w, 256), max(h, 256)

def parse_input_image(raw_data: str) -> Image.Image:
    if raw_data.startswith("http://") or raw_data.startswith("https://"):
        resp = requests.get(raw_data, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    
    if "," in raw_data:
        raw_data = raw_data.split(",", 1)[1]
    
    return Image.open(io.BytesIO(base64.b64decode(raw_data))).convert("RGB")

def save_driving_video_file(raw_video: str) -> str:
    """Saves raw video string (URL or base64) to a local temp file and returns path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    
    if raw_video.startswith("http://") or raw_video.startswith("https://"):
        resp = requests.get(raw_video, timeout=60)
        resp.raise_for_status()
        tmp.write(resp.content)
    else:
        if "," in raw_video:
            raw_video = raw_video.split(",", 1)[1]
        tmp.write(base64.b64decode(raw_video))
    
    tmp.close()
    return tmp.name

def inspect_driving_video(video_path: str):
    """Extracts FPS and total frames from the driving video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # Lock generation to 49 or 81 frames to fit Wan DiT temporal block constraints
    usable_frames = 81 if frame_count >= 81 else 49
    return int(fps), usable_frames

def handler(job):
    job_input = job.get("input", {})
    char_image_raw = job_input.get("character_image")
    driving_video_raw = job_input.get("driving_video")

    if not char_image_raw:
        return {"error": "Missing 'character_image' input."}
    if not driving_video_raw:
        return {"error": "Missing 'driving_video' input."}

    prompt = job_input.get("prompt", "High fidelity character animation matching driving motion, photorealistic.")
    negative_prompt = job_input.get("negative_prompt", "blurry, distorted anatomy, warped limbs, artifacts")
    steps = int(job_input.get("num_inference_steps", 30))
    target_w = int(job_input.get("width", 720))
    target_h = int(job_input.get("height", 1280))
    target_w, target_h = safe_dimensions(target_w, target_h)

    # 1. Decode Character Image
    character_img = parse_input_image(char_image_raw)
    character_img = character_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # 2. Extract Motion Properties from Driving Video
    temp_driving_path = save_driving_video_file(driving_video_raw)
    fps, num_frames = inspect_driving_video(temp_driving_path)
    if os.path.exists(temp_driving_path):
        os.remove(temp_driving_path)

    pipe = get_pipeline()
    generator = torch.Generator(device="cpu").manual_seed(42)

    start_t = time.time()
    with torch.inference_mode():
        output = pipe(
            image=character_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=target_h,
            width=target_w,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=5.0,
            generator=generator
        )

    frames = output.frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
        out_video_path = tmp_out.name

    export_to_video(frames, out_video_path, fps=fps)

    with open(out_video_path, "rb") as vf:
        encoded_b64 = base64.b64encode(vf.read()).decode("utf-8")

    os.remove(out_video_path)

    return {
        "status": "success",
        "video_base64": encoded_b64,
        "frames_rendered": num_frames,
        "fps": fps,
        "render_time_sec": round(time.time() - start_t, 2)
    }

runpod.serverless.start({"handler": handler})
