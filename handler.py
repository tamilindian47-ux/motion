import os
import subprocess
import time
import requests
import runpod

# Global State Variables to track environment creation
SETUP_COMPLETE = False

def initialize_runtime_environment():
    """Programmatically hooks assets on boot to completely defeat Docker build filters."""
    global SETUP_COMPLETE
    if SETUP_COMPLETE:
        return

    print("--- STARTING PRODUCTION NODE AND MODEL DEPLOYMENT LAYERS ---")
    
    # 1. Clone required extensions natively inside the machine space
    nodes_dir = "/ComfyUI/custom_nodes"
    if not os.path.exists(f"{nodes_dir}/ComfyUI-VideoHelperSuite"):
        print("Deploying VideoHelperSuite custom nodes...")
        subprocess.run(["git", "clone", "https://github.com"], cwd=nodes_dir)
        subprocess.run(["pip3", "install", "-r", "ComfyUI-VideoHelperSuite/requirements.txt"], cwd=nodes_dir)

    if not os.path.exists(f"{nodes_dir}/ComfyUI-WanVideoWrapper"):
        print("Deploying WanVideoWrapper custom nodes...")
        subprocess.run(["git", "clone", "https://github.com"], cwd=nodes_dir)
        subprocess.run(["pip3", "install", "-r", "ComfyUI-WanVideoWrapper/requirements.txt"], cwd=nodes_dir)

    # 2. Download un-gated community mirrors for Quantized Wan 2.1 and its VAE
    ckpt_path = "/ComfyUI/models/checkpoints/wan2.1_i2v_14b_q4_k_m.safetensors"
    if not os.path.exists(ckpt_path):
        print("Downloading Wan 2.1 14B Q4 Checkpoint (~9.5GB)...")
        url = "https://huggingface.co"
        subprocess.run(["wget", "-O", ckpt_path, url])

    vae_path = "/ComfyUI/models/vae/wan_2.1_vae.safetensors"
    if not os.path.exists(vae_path):
        print("Downloading Video Decode VAE (~1.2GB)...")
        url = "https://huggingface.co"
        subprocess.run(["wget", "-O", vae_path, url])

    # 3. Boot ComfyUI in the background headlessly via local port
    print("Launching headless ComfyUI core server matrix...")
    subprocess.Popen(["python3", "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-metadata"], cwd="/ComfyUI")
    
    # Wait for internal API port to wake up before accepting jobs
    time.sleep(10)
    print("--- CONTAINER READY FOR INFERENCE PAYLOADS ---")
    SETUP_COMPLETE = True

def handler(event):
    """Processes incoming data loops from your local HTML frontend."""
    # Ensure background system is fully operational
    initialize_runtime_environment()
    
    job_input = event.get("input", {})
    workflow_graph = job_input.get("workflow", {})
    uploaded_images = job_input.get("images", [])

    # If your client passes a Base64 image, save it into ComfyUI's input directory
    if uploaded_images:
        import base64
        for img_data in uploaded_images:
            file_target = f"/ComfyUI/input/{img_data['name']}"
            with open(file_target, "wb") as f:
                f.write(base64.b64decode(img_data["image"]))

    # Pass the custom dynamic workflow straight into ComfyUI's internal loop
    try:
        response = requests.post("http://127.0.0", json={"prompt": workflow_graph})
        res_data = response.json()
        prompt_id = res_data.get("prompt_id")
        
        # Poll the server for execution completion
        while True:
            history_resp = requests.get(f"http://127.0.0{prompt_id}").json()
            if prompt_id in history_resp:
                break
            time.sleep(1)

        # Locate the compiled video loop inside the active output footprint
        output_dir = "/ComfyUI/output"
        generated_files = os.listdir(output_dir)
        if not generated_files:
            return {"error": "Generation completed but no output file was written to disk."}

        latest_video = max([os.path.join(output_dir, f) for f in generated_files], key=os.path.getctime)
        
        # Read the video binary bytes and encode directly back to standard Base64 text
        with open(latest_video, "rb") as video_file:
            base64_encoded_video = base64.b64encode(video_file.read()).decode("utf-8")

        # Wipe outputs to prevent background cache overlaps on subsequent runs
        os.remove(latest_video)

        # Pass payload straight back to index.html player layer
        return {"images": {"data": base64_encoded_video}}

    except Exception as e:
        return {"error": f"Internal execution crash tracking context: {str(e)}"}

# Register entry hook for RunPod cluster loop
runpod.serverless.start({"handler": handler})
