import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import gdown
import onnx
from huggingface_hub import snapshot_download


ROOT = Path("/opt/rehiface")

REVISION = "89cf4639ae54c66302a97903efd09faaa71aa882"

MODEL_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1hVWFXPIDwACqoKKtgXAJubYC_H4k5njc"
)

MODEL_NAMES = [
    "9O_865k.onnx",
    "CurricularFace.tjm",
    "gfpganv14_fp32_bs1_scale.onnx",
    "pfpld_robust_sim_bs1_8003.onnx",
    "scrfd_500m_bnkps_shape640x640.onnx",
    "xseg_230611_16_17.onnx",
]


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    print("Downloading pinned ReHiFace-S source...", flush=True)

    snapshot_download(
        repo_id="GuijiAI/ReHiFace-S",
        revision=REVISION,
        local_dir=str(ROOT),
        ignore_patterns=[
            "assets/*",
            "data/*.mp4",
            "data/*.avi",
        ],
    )

    weights = ROOT / "pretrain_models"
    weights.mkdir(parents=True, exist_ok=True)

    print("Downloading official pretrained models...", flush=True)

    with tempfile.TemporaryDirectory() as temporary_directory:
        downloaded = gdown.download_folder(
            url=MODEL_FOLDER_URL,
            output=temporary_directory,
            quiet=False,
            remaining_ok=False,
        )

        if not downloaded:
            raise RuntimeError(
                "Official weights download failed. "
                "Google Drive access or quota may be blocking it."
            )

        download_root = Path(temporary_directory)

        for name in MODEL_NAMES:
            matches = list(download_root.rglob(name))

            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one {name}; found {len(matches)}."
                )

            source = matches[0]

            if source.stat().st_size < 100_000:
                raise RuntimeError(
                    f"Downloaded model is suspiciously small: {name}"
                )

            shutil.copy2(source, weights / name)

    manifest = {}

    for name in MODEL_NAMES:
        path = weights / name

        if path.suffix == ".onnx":
            print(f"Checking ONNX structure: {name}", flush=True)
            onnx.checker.check_model(onnx.load(str(path)))

        manifest[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    (ROOT / "MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    (ROOT / "SOURCE_REVISION.txt").write_text(
        REVISION,
        encoding="utf-8",
    )

    print(
        "Source and weights prepared. GPU inference has not been tested.",
        flush=True,
    )


if __name__ == "__main__":
    main()
