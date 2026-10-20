import os

# Disable HuggingFace's unstable Xet transfer backend to prevent background writer crashes
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import io
import uuid
import base64
import torch
from PIL import Image
import runpod
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download

STORAGE_ROOT = "/runpod-volume" if os.path.exists("/runpod-volume") else "/app/models_cache"
MODEL_DIR = os.path.join(STORAGE_ROOT, "ltx_video_model")

MODEL_ID = "Lightricks/LTX-Video"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIPELINE = None

def get_or_load_pipeline():
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    os.makedirs(MODEL_DIR, exist_ok=True)
    marker = os.path.join(MODEL_DIR, "model_index.json")

    # Download model weights using stable HTTP streaming
    if not os.path.exists(marker):
        print("Downloading LTX-Video weights via stable HTTP stream...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_DIR,
            max_workers=4
        )

    print("Loading LTXImageToVideoPipeline into GPU (bfloat16)...")
    PIPELINE = LTXImageToVideoPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16
    ).to(DEVICE)
    
    PIPELINE.enable_vae_slicing()
    return PIPELINE

def decode_base64_image(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    image_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def handler(event):
    job_input = event.get("input", {})
    
    image_b64 = job_input.get("image")
    prompt = job_input.get("prompt", "High quality cinematic motion, smooth camera pan, 4k detailed")
    negative_prompt = job_input.get("negative_prompt", "worst quality, blurry, jittery, distorted, cartoon, low resolution")
    num_inference_steps = int(job_input.get("num_inference_steps", 30))
    guidance_scale = float(job_input.get("guidance_scale", 3.0))
    num_frames = int(job_input.get("num_frames", 121))
    fps = int(job_input.get("fps", 24))
    width = int(job_input.get("width", 768))
    height = int(job_input.get("height", 512))
    seed = int(job_input.get("seed", 42))

    if not image_b64:
        return {"error": "Input 'image' (base64) is required."}

    job_id = str(uuid.uuid4())[:8]
    work_dir = f"/tmp/{job_id}"
    os.makedirs(work_dir, exist_ok=True)
    output_video_path = os.path.join(work_dir, "output.mp4")

    try:
        input_image = decode_base64_image(image_b64)
        
        # Dimensions must be divisible by 32 for LTX VAE
        width = (width // 32) * 32
        height = (height // 32) * 32
        input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)

        pipe = get_or_load_pipeline()

        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        output = pipe(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        video_frames = output.frames[0]
        export_to_video(video_frames, output_video_path, fps=fps)

        with open(output_video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "status": "success",
            "video_base64": video_b64,
            "filename": "ltx_video_output.mp4"
        }

    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}"}

    finally:
        if os.path.exists(work_dir):
            import shutil
            shutil.rmtree(work_dir)

runpod.serverless.start({"handler": handler})
