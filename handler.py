import os
import sys

# Force absolute working directory
os.chdir("/app/MimicMotion")
sys.path.insert(0, "/app/MimicMotion")

import uuid
import base64
import urllib.request
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import runpod

STORAGE_ROOT = "/runpod-volume" if os.path.exists("/runpod-volume") else "/app/models_cache"
MODELS_DIR = os.path.join(STORAGE_ROOT, "mimicmotion_models")
DWPOSE_DIR = os.path.join(MODELS_DIR, "DWPose")
SVD_DIR = os.path.join(MODELS_DIR, "SVD")
CKPT_PATH = os.path.join(MODELS_DIR, "MimicMotion_1-1.pth")

YOLOX_PATH = os.path.join(DWPOSE_DIR, "yolox_l.onnx")
DW_PATH = os.path.join(DWPOSE_DIR, "dw-ll_ucoco_384.onnx")

PIPELINE = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_assets():
    os.makedirs(DWPOSE_DIR, exist_ok=True)
    os.makedirs(SVD_DIR, exist_ok=True)
    
    # 1. Download DWPose ONNX files
    if not os.path.exists(YOLOX_PATH) or os.path.getsize(YOLOX_PATH) == 0:
        print("Downloading DWPose YOLOX...")
        urllib.request.urlretrieve("https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx", YOLOX_PATH)
        
    if not os.path.exists(DW_PATH) or os.path.getsize(DW_PATH) == 0:
        print("Downloading DWPose keypoints...")
        urllib.request.urlretrieve("https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx", DW_PATH)
        
    # 2. Download MimicMotion Checkpoint
    if not os.path.exists(CKPT_PATH) or os.path.getsize(CKPT_PATH) == 0:
        print("Downloading MimicMotion checkpoint...")
        urllib.request.urlretrieve("https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth", CKPT_PATH)
        
    # 3. Download SVD Base Model
    svd_marker = os.path.join(SVD_DIR, "model_index.json")
    if not os.path.exists(svd_marker):
        print("Downloading SVD XT 1.1 weights...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="vdo/stable-video-diffusion-img2vid-xt-1-1", local_dir=SVD_DIR)

def get_or_load_pipeline():
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE
        
    ensure_assets()
    
    from mimicmotion.modules.pose_net import PoseNet
    from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
    from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel, EulerDiscreteScheduler
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    
    print("Loading models to GPU...")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(SVD_DIR, subfolder="vae").to(DEVICE, dtype=torch.float16)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(SVD_DIR, subfolder="image_encoder").to(DEVICE, dtype=torch.float16)
    feature_extractor = CLIPImageProcessor.from_pretrained(SVD_DIR, subfolder="feature_extractor")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(SVD_DIR, subfolder="unet").to(DEVICE, dtype=torch.float16)
    scheduler = EulerDiscreteScheduler.from_pretrained(SVD_DIR, subfolder="scheduler")
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0]).to(DEVICE, dtype=torch.float16)
    
    state_dict = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    unet_dict = {k.replace("unet.", ""): v for k, v in state_dict.items() if k.startswith("unet.")}
    posenet_dict = {k.replace("pose_net.", ""): v for k, v in state_dict.items() if k.startswith("pose_net.")}
    
    if unet_dict:
        unet.load_state_dict(unet_dict, strict=False)
    if posenet_dict:
        pose_net.load_state_dict(posenet_dict, strict=False)
        
    PIPELINE = MimicMotionPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
        pose_net=pose_net
    )
    return PIPELINE

def decode_base64_file(b64_str: str, output_path: str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(b64_str))

def handler(event):
    job_input = event.get("input", {})
    ref_image_b64 = job_input.get("ref_image")
    motion_video_b64 = job_input.get("motion_video")
    num_inference_steps = int(job_input.get("num_inference_steps", 25))
    guidance_scale = float(job_input.get("guidance_scale", 2.0))
    fps = int(job_input.get("fps", 15))
    num_frames = int(job_input.get("num_frames", 72))
    resolution = int(job_input.get("resolution", 576))
    frames_overlap = int(job_input.get("frames_overlap", 6))
    sample_stride = int(job_input.get("sample_stride", 2))
    seed = int(job_input.get("seed", 42))

    if not ref_image_b64 or not motion_video_b64:
        return {"error": "Both 'ref_image' and 'motion_video' are required."}

    job_id = str(uuid.uuid4())[:8]
    work_dir = f"/tmp/{job_id}"
    os.makedirs(work_dir, exist_ok=True)
    input_image_path = os.path.join(work_dir, "ref_image.png")
    input_video_path = os.path.join(work_dir, "motion_video.mp4")
    output_video_path = os.path.join(work_dir, "output.mp4")

    try:
        decode_base64_file(ref_image_b64, input_image_path)
        decode_base64_file(motion_video_b64, input_video_path)

        pipeline = get_or_load_pipeline()
        
        # Override DWPose detector with absolute paths
        from mimicmotion.dwpose.preprocess import dwprocessor
        from mimicmotion.dwpose.wholebody import Wholebody
        dwprocessor.args = (YOLOX_PATH, DW_PATH, "cuda:0" if torch.cuda.is_available() else "cpu")
        dwprocessor.pose_estimation = Wholebody(*dwprocessor.args)
        
        from inference import preprocess
        from mimicmotion.utils.utils import save_to_mp4
        
        pose_pixels, image_pixels = preprocess(
            input_video_path,
            input_image_path,
            resolution=resolution,
            sample_stride=sample_stride
        )
        
        if pose_pixels.size(0) > num_frames:
            pose_pixels = pose_pixels[:num_frames]

        # Cast pose_pixels to float16 to match pose_net's FP16 weights
        pose_pixels = pose_pixels.to(dtype=torch.float16)

        pil_images = [TF.to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
        
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(seed)
        
        frames = pipeline(
            image=pil_images,
            image_pose=pose_pixels,
            num_frames=pose_pixels.size(0),
            tile_size=num_frames,
            tile_overlap=frames_overlap,
            height=pose_pixels.shape[-2],
            width=pose_pixels.shape[-1],
            fps=fps,
            noise_aug_strength=0.0,
            num_inference_steps=num_inference_steps,
            generator=generator,
            min_guidance_scale=guidance_scale,
            max_guidance_scale=guidance_scale,
            decode_chunk_size=8,
            output_type="pt",
            device=DEVICE
        ).frames.cpu()

        save_to_mp4(frames[0], output_video_path, fps=fps)

        with open(output_video_path, "rb") as f:
            final_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "status": "success",
            "video_base64": final_b64,
            "filename": "mimicmotion_output.mp4"
        }

    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}"}

    finally:
        if os.path.exists(work_dir):
            import shutil
            shutil.rmtree(work_dir)

runpod.serverless.start({"handler": handler})
