import os
import sys
import time
import base64
import tempfile
import subprocess
import traceback
from contextlib import contextmanager

import cv2
import numpy as np
import requests
import onnxruntime as ort
import insightface
import runpod


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MODEL_DIR = "/app/checkpoints"

OUTPUT_CRF = os.environ.get("OUTPUT_CRF", "17")
X264_PRESET = os.environ.get("X264_PRESET", "fast")

VALID_X264_PRESETS = {
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow",
}
if X264_PRESET not in VALID_X264_PRESETS:
    raise ValueError(f"Invalid X264_PRESET: {X264_PRESET}")

# 85% enhanced high-fidelity blend
ENHANCE_BLEND = float(os.environ.get("ENHANCE_BLEND", "0.85"))
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.40"))
DETECTION_THRESHOLD = float(os.environ.get("DETECTION_THRESHOLD", "0.35"))
LOG_EVERY_FRAMES = max(1, int(os.environ.get("LOG_EVERY_FRAMES", "25")))

cv2.setNumThreads(1)

print("=== Initialising professional face-swap worker ===", flush=True)

try:
    ort.preload_dlls()
except Exception as exc:
    print(f"ONNX DLL preload warning: {exc}", flush=True)

available = ort.get_available_providers()
print(f"Available ONNX providers: {available}", flush=True)

if "CUDAExecutionProvider" not in available:
    raise RuntimeError("CUDAExecutionProvider is unavailable.")

PROVIDERS = [
    (
        "CUDAExecutionProvider",
        {"cudnn_conv_algo_search": "HEURISTIC"},
    )
]

# ENABLED: landmark_3d_68 loads the 3D mesh model from buffalo_l
face_app = insightface.app.FaceAnalysis(
    name="buffalo_l",
    root="/root/.insightface",
    allowed_modules=["detection", "recognition", "landmark_3d_68"],
    providers=PROVIDERS,
)
face_app.prepare(
    ctx_id=0,
    det_thresh=DETECTION_THRESHOLD,
    det_size=(640, 640),
)

swapper = insightface.model_zoo.get_model(
    os.path.join(MODEL_DIR, "inswapper_128.onnx"),
    download=False,
    providers=PROVIDERS,
)

occ_session = ort.InferenceSession(
    os.path.join(MODEL_DIR, "face_occluder.onnx"),
    providers=PROVIDERS,
)

# BiSeNet Semantic Face Parser Session
parser_path = os.path.join(MODEL_DIR, "face_parser.onnx")
parser_session = None
if os.path.exists(parser_path):
    parser_session = ort.InferenceSession(parser_path, providers=PROVIDERS)
    print("BiSeNet Semantic Parser loaded successfully.", flush=True)

enh_session = None
enh_input_name = None
enh_size = None
enh_template = None

occ_input = occ_session.get_inputs()[0]
occ_shape = occ_input.shape
occ_nchw = len(occ_shape) == 4 and occ_shape[1] == 3

try:
    if occ_nchw:
        occ_size = (int(occ_shape[3]), int(occ_shape[2]))
    else:
        occ_size = (int(occ_shape[2]), int(occ_shape[1]))
except (TypeError, ValueError):
    occ_size = (256, 256)

mask_cache = {}


def check_cuda(label, session):
    active = session.get_providers()
    print(f"{label} providers: {active}", flush=True)
    if "CUDAExecutionProvider" not in active:
        raise RuntimeError(f"{label} does not have CUDA enabled.")


for model_name, model in face_app.models.items():
    check_cuda(model_name, model.session)

check_cuda("Swapper", swapper.session)
check_cuda("Occluder", occ_session)
if parser_session:
    check_cuda("FaceParser", parser_session)

print("=== Worker ready ===", flush=True)


# ------------------------------------------------------------
# 3D Rigid Alignment & Semantic Masking
# ------------------------------------------------------------

def stabilize_landmarks_with_3d(face):
    """
    Reconstructs stable 5 anchor points from 3D 68-point landmarks.
    Prevents the 2D affine matrix from collapsing during kissing/profile views.
    """
    lm3d = getattr(face, "landmark_3d_68", None)
    if lm3d is not None and len(lm3d) == 68:
        pts = lm3d.astype(np.float32)
        # Average the 6 vertices of each eye for rigid anchoring
        left_eye = np.mean(pts[36:42, :2], axis=0)
        right_eye = np.mean(pts[42:48, :2], axis=0)
        nose = pts[30, :2]
        mouth_l = pts[48, :2]
        mouth_r = pts[54, :2]
        # Replace the fragile 2D detector kps with the 3D-projected landmarks
        face.kps = np.array([left_eye, right_eye, nose, mouth_l, mouth_r], dtype=np.float32)


def semantic_parsing_mask(crop_bgr):
    """
    Uses BiSeNet to isolate face skin, eyes, nose, and lips.
    Prevents replacement pixels from bleeding onto the kissing partner's face.
    """
    if parser_session is None:
        return np.ones((crop_bgr.shape[0], crop_bgr.shape[1]), dtype=np.float32)

    h, w = crop_bgr.shape[:2]
    # BiSeNet expects 512x512 RGB normalized with ImageNet stats
    resized = cv2.resize(crop_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (rgb - mean) / std
    tensor = norm.transpose(2, 0, 1)[None].astype(np.float32)

    input_name = parser_session.get_inputs()[0].name
    out = parser_session.run(None, {input_name: tensor})[0]
    parsing = np.argmax(out[0], axis=0)

    # 1:skin, 2:l_eyebrow, 3:r_eyebrow, 4:l_eye, 5:r_eye, 10:nose, 11:mouth, 12:upper_lip, 13:lower_lip
    target_classes = [1, 2, 3, 4, 5, 10, 11, 12, 13]
    binary_mask = np.isin(parsing, target_classes).astype(np.float32)
    binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(binary_mask, (0, 0), 3.0)


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def normalise(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError("Invalid face embedding.")
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError("Face embedding has zero length.")
    return vector / length


def resolve_input_to_file(data_or_url, out_path):
    if not isinstance(data_or_url, str) or not data_or_url:
        raise ValueError("Media must be a URL or base64 string.")

    if data_or_url.startswith(("https://", "http://")):
        with requests.get(data_or_url, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            with open(out_path, "wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    else:
        encoded = data_or_url
        if encoded.startswith("data:"):
            encoded = encoded.split(",", 1)[1]
        with open(out_path, "wb") as handle:
            handle.write(base64.b64decode(encoded))


def decode_image(data_or_url, directory, index):
    path = os.path.join(directory, f"source_{index}.img")
    resolve_input_to_file(data_or_url, path)
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Cannot decode replacement image {index + 1}.")
    return image


@contextmanager
def timed(stats, stage):
    started = time.perf_counter()
    try:
        yield
    finally:
        stats[stage] = stats.get(stage, 0.0) + (time.perf_counter() - started)


def box_mask(size, blur=0.2):
    key = (size, blur)
    if key not in mask_cache:
        amount = int(size * 0.2 * blur)
        edge = max(amount // 2, 1)
        mask = np.ones((size, size), dtype=np.float32)
        mask[:edge, :] = 0
        mask[-edge:, :] = 0
        mask[:, :edge] = 0
        mask[:, -edge:] = 0
        mask_cache[key] = cv2.GaussianBlur(mask, (0, 0), max(amount * 0.2, 0.1))
    return mask_cache[key]


def occlusion_mask(crop_bgr):
    image = cv2.resize(crop_bgr, occ_size).astype(np.float32) / 255.0
    if occ_nchw:
        image = image.transpose(2, 0, 1)
    tensor = np.ascontiguousarray(image[None])
    output = occ_session.run(None, {occ_input.name: tensor})[0][0]
    mask = np.clip(np.squeeze(output), 0, 1).astype(np.float32)
    mask = cv2.resize(mask, (crop_bgr.shape[1], crop_bgr.shape[0]))
    mask = cv2.GaussianBlur(mask, (0, 0), 3)
    mask = (np.clip(mask, 0.4, 1.0) - 0.4) * 1.6
    return np.clip(mask, 0, 1)


def paste_crop(frame, crop, mask, matrix):
    inverse = cv2.invertAffineTransform(matrix)
    height, width = crop.shape[:2]
    corners = np.array([[[0, 0], [width, 0], [width, height], [0, height]]], dtype=np.float32)
    points = cv2.transform(corners, inverse)[0]

    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, int(np.floor(points[:, 0].min())) - 2)
    y1 = max(0, int(np.floor(points[:, 1].min())) - 2)
    x2 = min(frame_w, int(np.ceil(points[:, 0].max())) + 2)
    y2 = min(frame_h, int(np.ceil(points[:, 1].max())) + 2)

    if x2 <= x1 or y2 <= y1:
        return frame

    local_inverse = inverse.copy()
    local_inverse[0, 2] -= x1
    local_inverse[1, 2] -= y1
    size = (x2 - x1, y2 - y1)

    # High-quality bicubic resampling
    warped_crop = cv2.warpAffine(
        crop, local_inverse, size,
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(np.float32)

    warped_mask = cv2.warpAffine(
        mask, local_inverse, size,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    alpha = np.clip(warped_mask, 0, 1)[:, :, None]
    original = frame[y1:y2, x1:x2].astype(np.float32)

    frame[y1:y2, x1:x2] = np.clip(
        original * (1.0 - alpha) + warped_crop * alpha, 0, 255
    ).astype(np.uint8)

    return frame


# ------------------------------------------------------------
# Swap and Enhancement
# ------------------------------------------------------------

def swap_face(frame, face, source_face, stats):
    # Anchor landmarks using 3D mesh before computing warp matrix
    stabilize_landmarks_with_3d(face)

    with timed(stats, "swap"):
        fake, matrix = swapper.get(frame, face, source_face, paste_back=False)

    if fake is None:
        return frame

    fake = np.ascontiguousarray(fake)
    size = fake.shape[0]

    with timed(stats, "occlusion"):
        original_crop = cv2.warpAffine(frame, matrix, (size, size), borderValue=0)
        occ_m = occlusion_mask(original_crop)
        parse_m = semantic_parsing_mask(original_crop)
        
        # Combine box boundary, generic occluder, and anatomical parsing
        mask = np.minimum(np.minimum(box_mask(size), occ_m), parse_m)

    with timed(stats, "composite"):
        return paste_crop(frame, fake, mask, matrix)


def load_enhancer():
    global enh_session, enh_input_name, enh_size, enh_template
    if enh_session is not None:
        return

    print("Loading GPEN enhancer...", flush=True)
    enh_session = ort.InferenceSession(os.path.join(MODEL_DIR, "enhancer.onnx"), providers=PROVIDERS)
    check_cuda("Enhancer", enh_session)

    model_input = enh_session.get_inputs()[0]
    enh_input_name = model_input.name
    enh_size = int(model_input.shape[2])

    enh_template = np.array(
        [
            [0.37691676, 0.46864664],
            [0.62285697, 0.46912813],
            [0.50123859, 0.61331904],
            [0.39308822, 0.72541100],
            [0.61150205, 0.72490465],
        ],
        dtype=np.float32,
    ) * enh_size


def enhance_face(swapped, original, face):
    blend = float(np.clip(ENHANCE_BLEND, 0, 1))
    if blend == 0:
        return swapped

    stabilize_landmarks_with_3d(face)

    matrix, _ = cv2.estimateAffinePartial2D(face.kps.astype(np.float32), enh_template, method=cv2.LMEDS)
    if matrix is None:
        return swapped

    crop = cv2.warpAffine(swapped, matrix, (enh_size, enh_size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    original_crop = cv2.warpAffine(original, matrix, (enh_size, enh_size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    tensor = crop[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])

    result = enh_session.run(None, {enh_input_name: tensor})[0][0]
    result = (np.clip(result, -1, 1) + 1.0) * 127.5
    result = np.rint(result.transpose(1, 2, 0))[:, :, ::-1]

    mixed = crop.astype(np.float32) * (1.0 - blend) + result.astype(np.float32) * blend

    parse_m = semantic_parsing_mask(original_crop)
    mask = np.minimum(np.minimum(box_mask(enh_size, 0.15), occlusion_mask(original_crop)), parse_m)

    return paste_crop(swapped, mixed, mask, matrix)


def apply_swaps(frame, targets, enhance, stats):
    with timed(stats, "detect_recognise"):
        faces = face_app.get(frame)

    swapped_count = 0
    for face in faces:
        embedding = face.normed_embedding
        similarities = [float(np.dot(embedding, target["target_embedding"])) for target in targets]
        best_index = int(np.argmax(similarities))

        if similarities[best_index] < MATCH_THRESHOLD:
            continue

        original = frame.copy() if enhance else None
        frame = swap_face(frame, face, targets[best_index]["source_face"], stats)

        if enhance:
            with timed(stats, "enhance"):
                frame = enhance_face(frame, original, face)

        swapped_count += 1

    return frame, swapped_count


# ------------------------------------------------------------
# Face Scanning & Clustering
# ------------------------------------------------------------

def collect_faces(frame, detections, min_size=0):
    if frame is None:
        return
    for face in face_app.get(frame):
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x2 - x1 <= min_size or y2 - y1 <= min_size:
            continue

        detections.append({
            "crop": frame[y1:y2, x1:x2].copy(),
            "embedding": face.normed_embedding,
            "score": (x2 - x1) * (y2 - y1),
        })


def cluster_embeddings(detections):
    clusters = []
    for index, detection in enumerate(detections):
        for cluster in clusters:
            centroid = normalise(np.mean([detections[i]["embedding"] for i in cluster], axis=0))
            if float(np.dot(detection["embedding"], centroid)) >= 0.58:
                cluster.append(index)
                break
        else:
            clusters.append([index])

    characters = []
    for character_id, cluster in enumerate(clusters):
        best = max(cluster, key=lambda i: detections[i]["score"])
        centroid = normalise(np.mean([detections[i]["embedding"] for i in cluster], axis=0))
        ok, buffer = cv2.imencode(".jpg", detections[best]["crop"])
        if not ok:
            continue
        characters.append({
            "character_id": character_id,
            "thumbnail_base64": base64.b64encode(buffer.tobytes()).decode("ascii"),
            "embedding": centroid.tolist(),
        })
    return characters


def detect_unique_people(path, animated):
    detections = []
    if not animated:
        image = cv2.imread(path)
        if image is None:
            raise ValueError("Cannot decode target image.")
        collect_faces(image, detections)
    else:
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                raise ValueError("Cannot open target video/GIF.")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
            step = max(1, total // 30)
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % step == 0:
                    collect_faces(frame, detections, min_size=40)
                index += 1
        finally:
            cap.release()
    return cluster_embeddings(detections)


# ------------------------------------------------------------
# Output Video Encoding
# ------------------------------------------------------------

def ffmpeg_error(log_file):
    log_file.flush()
    log_file.seek(0)
    return log_file.read().decode("utf-8", errors="replace")[-6000:]


def encode_video(path, output_path, targets, enhance, stats):
    cap = cv2.VideoCapture(path)
    proc = None
    count = 0
    swapped_count = 0
    started = time.perf_counter()

    try:
        if not cap.isOpened():
            raise ValueError("Cannot open input video.")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0 or fps > 120:
            fps = 25.0

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok:
            raise ValueError("Cannot read the first video frame.")

        height, width = frame.shape[:2]
        width -= width % 2
        height -= height % 2

        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "-", "-i", path,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-preset", X264_PRESET, "-crf", OUTPUT_CRF,
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", output_path,
        ]

        with tempfile.TemporaryFile() as error_log:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_log)
            try:
                while True:
                    result, changed = apply_swaps(frame, targets, enhance, stats)
                    swapped_count += changed

                    proc.stdin.write(np.ascontiguousarray(result[:height, :width]).tobytes())
                    count += 1

                    if count % LOG_EVERY_FRAMES == 0:
                        elapsed = time.perf_counter() - started
                        rate = count / max(elapsed, 1e-6)
                        eta = f"{(total - count) / rate:.1f}s" if total > count else "unknown"
                        print(f"PROGRESS: frame={count}/{total} fps={rate:.2f} eta={eta} swapped={swapped_count}", flush=True)

                    with timed(stats, "decode"):
                        ok, frame = cap.read()
                    if not ok:
                        break

            except (BrokenPipeError, OSError) as exc:
                if proc.poll() is None:
                    proc.terminate()
                details = ffmpeg_error(error_log)
                raise RuntimeError(f"FFmpeg pipe failed: {exc}\n{details}") from exc
            finally:
                if proc.stdin and not proc.stdin.closed:
                    try:
                        proc.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass

            if proc.wait(timeout=180) != 0:
                raise RuntimeError(f"FFmpeg error:\n{ffmpeg_error(error_log)}")

    finally:
        cap.release()

    return count, swapped_count


def execute_swap(path, media_type, targets, pipeline, directory):
    enhance = pipeline != "fast"
    if enhance:
        load_enhancer()

    stats = {}
    started = time.perf_counter()

    if media_type == "image":
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError("Cannot decode target image.")
        result, changed = apply_swaps(frame, targets, enhance, stats)
        ok, buffer = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 97])
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        frames = 1
    else:
        output_path = os.path.join(directory, "output.mp4")
        frames, changed = encode_video(path, output_path, targets, enhance, stats)

        if media_type == "gif":
            gif_path = os.path.join(directory, "output.gif")
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", output_path, "-filter_complex",
                "split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5",
                "-loop", "0", gif_path
            ], check=True)
            output_path = gif_path

        with open(output_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")

    total_seconds = time.perf_counter() - started
    timings = {k: round(v, 3) for k, v in stats.items()}
    print(f"STAGE TIMINGS: {timings}", flush=True)

    return {
        "status": "success",
        "media_type": media_type,
        "base64": encoded,
        "metrics": {
            "frames": frames,
            "swapped_face_instances": changed,
            "processing_seconds": round(total_seconds, 3),
            "stage_seconds": timings,
        },
    }


# ------------------------------------------------------------
# RunPod Entry Point
# ------------------------------------------------------------

def handler(event):
    job_id = event.get("id", "unknown")
    job = event.get("input", {})

    print(f"JOB START: id={job_id}, action={job.get('action', 'swap')}", flush=True)

    try:
        media = job.get("media")
        if not media:
            raise ValueError("Missing 'media' payload.")

        media_type = "gif" if job.get("is_gif") else ("video" if job.get("is_video", True) else "image")
        extension = "gif" if media_type == "gif" else ("mp4" if media_type == "video" else "jpg")
        action = job.get("action", "swap")

        with tempfile.TemporaryDirectory(prefix="face_swap_") as directory:
            path = os.path.join(directory, f"input.{extension}")
            resolve_input_to_file(media, path)

            if action == "detect":
                characters = detect_unique_people(path, animated=media_type != "image")
                return {"status": "success", "characters": characters}

            if action != "swap":
                raise ValueError(f"Unknown action: {action}")

            swaps = job.get("swaps", [])
            if not swaps:
                raise ValueError("No replacement faces were selected.")

            targets = []
            for index, spec in enumerate(swaps):
                source_image = decode_image(spec["new_face"], directory, index)
                source_faces = face_app.get(source_image)

                if not source_faces:
                    raise ValueError(f"No face detected in replacement image {index + 1}.")
                if len(source_faces) > 1:
                    raise ValueError(f"Replacement image {index + 1} contains multiple faces.")

                embedding = normalise(spec["target_embedding"])
                targets.append({"target_embedding": embedding, "source_face": source_faces[0]})

            return execute_swap(path, media_type, targets, job.get("pipeline", "gfpgan"), directory)

    except Exception as exc:
        traceback.print_exc()
        return {"status": "failed", "error": str(exc)}

    finally:
        print(f"JOB END: id={job_id}", flush=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
