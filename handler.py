import os
import io
import time
import base64
import tempfile
import requests
import torch
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
        print(f"Downloading model {MODEL_REPO} to {LOCAL_DIR}...")
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True
        )

    print("Initializing pipeline modules...")
    # Load VAE in fp32 to prevent numeric underflow/black frames
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

    # Enables automatic memory management across 24GB VRAM cards
    pipe.enable_model_cpu_offload()

    PIPELINE = pipe
    return PIPELINE

def safe_dimensions(width: int, height: int):
    # Wan requires spatial dimensions divisible by 16
    w = (int(width) // 16) * 16
    h = (int(height) // 16) * 16
    return max(w, 256), max(h, 256)

def parse_image(raw_data: str) -> Image.Image:
    if raw_data.startswith("http://") or raw_data.startswith("https://"):
        resp = requests.get(raw_data, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    
    if "," in raw_data:
        raw_data = raw_data.split(",", 1)[1]
    
    return Image.open(io.BytesIO(base64.b64decode(raw_data))).convert("RGB")

def handler(job):
    job_input = job.get("input", {})
    image_raw = job_input.get("image")
    if not image_raw:
        return {"error": "Missing 'image' parameter (base64 or URL)."}

    prompt = job_input.get("prompt", "Realistic natural motion, highly detailed character performance.")
    negative_prompt = job_input.get("negative_prompt", "blurry, distorted anatomy, warped limbs, low quality")
    
    steps = int(job_input.get("num_inference_steps", 30))
    guidance = float(job_input.get("guidance_scale", 5.0))
    seed = int(job_input.get("seed", 42))
    num_frames = int(job_input.get("num_frames", 49))
    fps = int(job_input.get("fps", 16))

    target_width = int(job_input.get("width", 720))
    target_height = int(job_input.get("height", 1280))
    target_width, target_height = safe_dimensions(target_width, target_height)

    source_img = parse_image(image_raw)
    source_img = source_img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    pipe = get_pipeline()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    start_t = time.time()
    with torch.inference_mode():
        output = pipe(
            image=source_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=target_height,
            width=target_width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        )

    frames = output.frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        out_video_path = tmp_file.name

    export_to_video(frames, out_video_path, fps=fps)

    with open(out_video_path, "rb") as vf:
        encoded_b64 = base64.b64encode(vf.read()).decode("utf-8")

    os.remove(out_video_path)

    return {
        "status": "success",
        "video_base64": encoded_b64,
        "width": target_width,
        "height": target_height,
        "frames": num_frames,
        "render_time_sec": round(time.time() - start_t, 2)
    }

runpod.serverless.start({"handler": handler})
