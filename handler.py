import os
import sys
import uuid
import base64
import urllib.request
import torch
import numpy as np
from PIL import Image
import runpod

sys.path.insert(0, "/app/MimicMotion")

STORAGE_ROOT = "/runpod-volume" if os.path.exists("/runpod-volume") else "/app/models_cache"
MODELS_DIR = os.path.join(STORAGE_ROOT, "mimicmotion_models")
DWPOSE_DIR = os.path.join(MODELS_DIR, "DWPose")
SVD_DIR = os.path.join(MODELS_DIR, "SVD")
CKPT_PATH = os.path.join(MODELS_DIR, "MimicMotion_1-1.pth")

REPO_MODELS_DIR = "/app/MimicMotion/models"

# Global pipeline holder for zero-overhead warm re-use
PIPELINE = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_assets():
    os.makedirs(DWPOSE_DIR, exist_ok=True)
    os.makedirs(SVD_DIR, exist_ok=True)
    
    # DWPose weights
    yolox_path = os.path.join(DWPOSE_DIR, "yolox_l.onnx")
    if not os.path.exists(yolox_path) or os.path.getsize(yolox_path) == 0:
        print("Downloading DWPose YOLOX...")
        urllib.request.urlretrieve("https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx", yolox_path)
        
    dw_path = os.path.join(DWPOSE_DIR, "dw-ll_ucoco_384.onnx")
    if not os.path.exists(dw_path) or os.path.getsize(dw_path) == 0:
        print("Downloading DWPose keypoints...")
        urllib.request.urlretrieve("https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx", dw_path)
        
    # MimicMotion Checkpoint
    if not os.path.exists(CKPT_PATH) or os.path.getsize(CKPT_PATH) == 0:
        print("Downloading MimicMotion checkpoint...")
        urllib.request.urlretrieve("https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth", CKPT_PATH)
        
    # SVD Base Model
    svd_marker = os.path.join(SVD_DIR, "model_index.json")
    if not os.path.exists(svd_marker):
        print("Downloading SVD XT 1.1 weights...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="vdo/stable-video-diffusion-img2vid-xt-1-1", local_dir=SVD_DIR)

    # Symlink for DWPose detector
    os.makedirs(REPO_MODELS_DIR, exist_ok=True)
    repo_dwpose = os.path.join(REPO_MODELS_DIR, "DWPose")
    if not os.path.exists(repo_dwpose):
        os.symlink(DWPOSE_DIR, repo_dwpose)

def get_or_load_pipeline():
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE
        
    ensure_assets()
    
    from mimicmotion.modules.pose_net import PoseNet
    from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
    from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel, EulerDiscreteScheduler
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    
    print("Loading base SVD models into GPU...")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(SVD_DIR, subfolder="vae").to(DEVICE, dtype=torch.float16)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(SVD_DIR, subfolder="image_encoder").to(DEVICE, dtype=torch.float16)
    feature_extractor = CLIPImageProcessor.from_pretrained(SVD_DIR, subfolder="feature_extractor")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(SVD_DIR, subfolder="unet").to(DEVICE, dtype=torch.float16)
    scheduler = EulerDiscreteScheduler.from_pretrained(SVD_DIR, subfolder="scheduler")
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0]).to(DEVICE, dtype=torch.float16)
    
    print("Loading MimicMotion weights...")
    state_dict = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    
    # Load weights into unet and pose_net directly
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
        
        from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose
        from mimicmotion.utils.utils import save_to_mp4
        
        # 1. Process Input Image and Poses directly
        ref_image = Image.open(input_image_path).convert("RGB")
        image_pixels = np.array(ref_image)
        
        # Extract pose sequence from motion video
        pose_pixels = get_video_pose(
            input_video_path,
            image_pixels,
            sample_stride=sample_stride,
            resolution=resolution
        )
        
        # Clip frames to requested length
        if len(pose_pixels) > num_frames:
            pose_pixels = pose_pixels[:num_frames]

        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        
        # 2. Run diffusion inference directly in PyTorch
        print("Running motion diffusion inference...")
        frames = pipeline(
            ref_image,
            pose_pixels,
            num_frames=len(pose_pixels),
            tile_size=num_frames,
            tile_overlap=frames_overlap,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="np"
        ).frames

        # 3. Save directly to MP4
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
