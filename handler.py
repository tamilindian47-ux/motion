import os
import sys
import uuid
import base64
import subprocess
import runpod

sys.path.insert(0, "/app/MimicMotion")

def decode_base64_file(b64_str: str, output_path: str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    data = base64.b64decode(b64_str)
    with open(output_path, "wb") as f:
        f.write(data)

def encode_file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def handler(event):
    job_input = event.get("input", {})
    
    ref_image_b64 = job_input.get("ref_image")
    motion_video_b64 = job_input.get("motion_video")
    num_inference_steps = int(job_input.get("num_inference_steps", 25))
    guidance_scale = float(job_input.get("guidance_scale", 2.0))
    fps = int(job_input.get("fps", 15))
    
    if not ref_image_b64 or not motion_video_b64:
        return {"error": "Both 'ref_image' and 'motion_video' are required."}
    
    job_id = str(uuid.uuid4())[:8]
    work_dir = f"/tmp/{job_id}"
    os.makedirs(work_dir, exist_ok=True)
    
    input_image_path = os.path.join(work_dir, "ref_image.png")
    input_video_path = os.path.join(work_dir, "motion_video.mp4")
    config_path = os.path.join(work_dir, "config.yaml")
    output_dir = os.path.join(work_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        decode_base64_file(ref_image_b64, input_image_path)
        decode_base64_file(motion_video_b64, input_video_path)
        
        yaml_content = f"""
base_model_path: "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
ckpt_path: "/app/MimicMotion/models/MimicMotion_1-1.pth"
dwpose_dir: "/app/MimicMotion/models/DWPose"
test_case:
  - ref_image_path: "{input_image_path}"
    ref_video_path: "{input_video_path}"
    num_inference_steps: {num_inference_steps}
    guidance_scale: {guidance_scale}
    fps: {fps}
"""
        with open(config_path, "w") as f:
            f.write(yaml_content)
        
        cmd = [
            "python", "inference.py",
            "--inference_config", config_path,
            "--output_dir", output_dir
        ]
        
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/app/MimicMotion",
            env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256")
        )
        
        if proc.returncode != 0:
            return {"error": f"Inference error: {proc.stderr}\nOutput: {proc.stdout}"}
            
        generated_files = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith(".mp4")
        ]
        
        if not generated_files:
            return {"error": "Inference completed but output video file was not generated."}
            
        final_video_b64 = encode_file_to_base64(generated_files[0])
        return {
            "status": "success",
            "video_base64": final_video_b64,
            "filename": "mimicmotion_output.mp4"
        }
        
    except Exception as e:
        return {"error": str(e)}
        
    finally:
        if os.path.exists(work_dir):
            subprocess.run(["rm", "-rf", work_dir])

runpod.serverless.start({"handler": handler})
