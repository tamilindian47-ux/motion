import os

# Disable Hugging Face Xet backend to prevent writer timeouts
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
MODEL_DIR = os.path.join(STORAGE_ROOT, "ltx_video_diffusers")

MODEL_ID = "Lightricks/LTX-Video"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIPELINE = None

DIFFUSERS_ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "transformer/*",
    "vae/*"
]

def get_or_load_pipeline():
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    os.makedirs(MODEL_DIR, exist_ok=True)
    marker = os.path.join(MODEL_DIR, "model_index.json")

    # Download model weights to storage once if not present
    if not os.path.exists(marker):
        print("Downloading LTX-Video diffusers weights (~12GB)...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_DIR,
            allow_patterns=DIFFUSERS_ALLOW_PATTERNS,
            max_workers=4
        )

    print("Loading LTXImageToVideoPipeline into GPU (bfloat16)...")
    PIPELINE = LTXImageToVideoPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16
    ).to(DEVICE)
    
    # Enable VAE tiling for low-VRAM decoding safety
    if hasattr(PIPELINE, "vae") and hasattr(PIPELINE.vae, "enable_tiling"):
        PIPELINE.vae.enable_tiling()
        
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
    
    duration_seconds = int(job_input.get("duration_seconds", 15))
    num_segments = max(1, min(4, duration_seconds // 5))  # Default 3 segments for 15s
    
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
        current_image = decode_base64_image(image_b64)
        
        # Dimensions must be divisible by 32 for LTX VAE
        width = (width // 32) * 32
        height = (height // 32) * 32
        current_image = current_image.resize((width, height), Image.Resampling.LANCZOS)

        pipe = get_or_load_pipeline()

        all_stitched_frames = []
        frames_per_segment = 121  # ~5.04 seconds @ 24fps

        for segment_idx in range(num_segments):
            print(f"Generating segment {segment_idx + 1}/{num_segments}...")
            
            segment_seed = seed + (segment_idx * 100)
            generator = torch.Generator(device=DEVICE).manual_seed(segment_seed)

            output = pipe(
                image=current_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=frames_per_segment,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )

            segment_frames = output.frames[0]
            
            if segment_idx == 0:
                all_stitched_frames.extend(segment_frames)
            else:
                all_stitched_frames.extend(segment_frames[1:])

            last_frame = segment_frames[-1]
            if isinstance(last_frame, Image.Image):
                current_image = last_frame
            else:
                import numpy as np
                current_image = Image.fromarray((last_frame * 255).astype(np.uint8))

        print(f"Exporting {len(all_stitched_frames)} total frames ({len(all_stitched_frames)/fps:.1f}s) to MP4...")
        export_to_video(all_stitched_frames, output_video_path, fps=fps)

        with open(output_video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "status": "success",
            "video_base64": video_b64,
            "filename": f"ltx_video_{duration_seconds}s.mp4",
            "total_frames": len(all_stitched_frames),
            "duration_seconds": round(len(all_stitched_frames) / fps, 2)
        }

    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}"}

    finally:
        if os.path.exists(work_dir):
            import shutil
            shutil.rmtree(work_dir)

runpod.serverless.start({"handler": handler})
