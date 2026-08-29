"""Benchmark the official EchoMimicV3-Flash pipeline on Modal."""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import modal


APP_NAME = "tightstudio-echomimic-v3-flash"
CACHE_PATH = Path("/cache")
BASE_PATH = CACHE_PATH / "Wan2.1-Fun-V1.1-1.3B-InP"
FLASH_PATH = CACHE_PATH / "EchoMimicV3" / "echomimicv3-flash-pro"
WAV2VEC_PATH = CACHE_PATH / "chinese-wav2vec2-base"
REPOSITORY_PATH = Path("/opt/echomimic_v3")
REPOSITORY_COMMIT = "7e89489ca51c0d008fc1963ec6c03fc5bd0b9397"
BASE_REVISION = "fc913c34361f4ec879e2f9c78b4f11ae50a937d1"
FLASH_REVISION = "311e176905a8c4c24b240b530488fe636ce4d249"
WAV2VEC_REVISION = "3991242c806928916fff4a8c0e4f76acf661b743"

BASE_FILES = [
  "Wan2.1_VAE.pth",
  "config.json",
  "configuration.json",
  "diffusion_pytorch_model.safetensors",
  "google/umt5-xxl/*",
  "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
  "models_t5_umt5-xxl-enc-bf16.pth",
  "xlm-roberta-large/*",
]
WAV2VEC_FILES = [
  "config.json",
  "preprocessor_config.json",
  "pytorch_model.bin",
]

cache_volume = modal.Volume.from_name(
  "tightstudio-echomimic-v3-flash-cache",
  create_if_missing=True,
)

image = (
  modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04",
    add_python="3.10",
  )
  .apt_install(
    "build-essential",
    "ffmpeg",
    "git",
    "libgl1",
    "libglib2.0-0",
    "libsndfile1",
  )
  .env(
    {
      "CUDA_HOME": "/usr/local/cuda",
      "HF_HOME": str(CACHE_PATH / "huggingface"),
      "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
      "TORCH_CUDA_ARCH_LIST": "8.9;9.0",
    }
  )
  .run_commands(
    "python -m pip install --upgrade pip wheel setuptools",
    "python -m pip install torch==2.6.0 torchvision==0.21.0 "
    "torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124",
    f"git clone https://github.com/antgroup/echomimic_v3.git {REPOSITORY_PATH}",
    f"git -C {REPOSITORY_PATH} checkout {REPOSITORY_COMMIT}",
    "python -m pip install Pillow einops safetensors timm torchdiffeq "
    "torchsde decord 'numpy<2' scikit-image opencv-python omegaconf "
    "SentencePiece 'imageio[ffmpeg]' ftfy accelerate diffusers==0.35.2 "
    "transformers==4.57.3 moviepy==2.2.1 librosa mmgp pyloudnorm",
  )
)

app = modal.App(APP_NAME, image=image)


class GpuMemoryMonitor:
  """Poll total GPU memory during inference."""

  def __init__(self) -> None:
    self.peak_mib = 0
    self._stop_event = threading.Event()
    self._thread = threading.Thread(target=self._poll, daemon=True)

  def start(self) -> None:
    self._thread.start()

  def stop(self) -> None:
    self._stop_event.set()
    self._thread.join(timeout=2)

  def _poll(self) -> None:
    while not self._stop_event.wait(0.5):
      result = subprocess.run(
        [
          "nvidia-smi",
          "--query-gpu=memory.used",
          "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
      )
      if result.returncode == 0:
        values = [
          int(value.strip())
          for value in result.stdout.splitlines()
          if value.strip().isdigit()
        ]
        if values:
          self.peak_mib = max(self.peak_mib, max(values))


def directory_size(path: Path) -> int:
  return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


@app.function(
  cpu=8,
  memory=32768,
  timeout=2 * 60 * 60,
  volumes={str(CACHE_PATH): cache_volume},
)
def prepare_weights() -> dict[str, Any]:
  """Cache the base, Flash adapter, and audio encoder."""
  from huggingface_hub import snapshot_download

  started_at = time.monotonic()
  snapshot_download(
    repo_id="alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP",
    revision=BASE_REVISION,
    local_dir=BASE_PATH,
    allow_patterns=BASE_FILES,
  )
  snapshot_download(
    repo_id="BadToBest/EchoMimicV3",
    revision=FLASH_REVISION,
    local_dir=FLASH_PATH.parent,
    allow_patterns=["echomimicv3-flash-pro/*"],
  )
  snapshot_download(
    repo_id="TencentGameMate/chinese-wav2vec2-base",
    revision=WAV2VEC_REVISION,
    local_dir=WAV2VEC_PATH,
    allow_patterns=WAV2VEC_FILES,
  )
  cache_volume.commit()
  return {
    "elapsed_seconds": round(time.monotonic() - started_at, 2),
    "cache_bytes": directory_size(CACHE_PATH),
    "base_bytes": directory_size(BASE_PATH),
    "flash_bytes": directory_size(FLASH_PATH),
    "wav2vec_bytes": directory_size(WAV2VEC_PATH),
  }


@app.function(
  gpu="L40S",
  cpu=12,
  memory=65536,
  timeout=3 * 60 * 60,
  max_containers=1,
  volumes={str(CACHE_PATH): cache_volume},
)
def generate_video(
  image_bytes: bytes,
  audio_bytes: bytes,
  prompt: str,
) -> dict[str, Any]:
  """Generate one 8-step Flash talking-avatar clip."""
  started_at = time.monotonic()
  with tempfile.TemporaryDirectory(prefix="echomimic-v3-flash-") as directory:
    work_path = Path(directory)
    image_path = work_path / "reference.jpg"
    audio_path = work_path / "speech.wav"
    output_directory = work_path / "output"
    output_path = output_directory / "reference_output.mp4"
    image_path.write_bytes(image_bytes)
    audio_path.write_bytes(audio_bytes)

    command = [
      "python",
      str(REPOSITORY_PATH / "infer_flash.py"),
      "--config_path",
      str(REPOSITORY_PATH / "config/config.yaml"),
      "--model_name",
      str(BASE_PATH),
      "--transformer_path",
      str(FLASH_PATH / "diffusion_pytorch_model.safetensors"),
      "--wav2vec_model_dir",
      str(WAV2VEC_PATH),
      "--save_path",
      str(output_directory),
      "--image_path",
      str(image_path),
      "--audio_path",
      str(audio_path),
      "--prompt",
      prompt,
      "--video_length",
      "93",
      "--num_inference_steps",
      "8",
      "--guidance_scale",
      "4",
      "--audio_guidance_scale",
      "2",
      "--sample_size",
      "768",
      "768",
      "--seed",
      "43",
    ]

    monitor = GpuMemoryMonitor()
    log_tail: deque[str] = deque(maxlen=220)
    monitor.start()
    try:
      process = subprocess.Popen(
        command,
        cwd=REPOSITORY_PATH,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
      )
      assert process.stdout is not None
      for line in process.stdout:
        print(line, end="", flush=True)
        log_tail.append(line.rstrip())
      return_code = process.wait()
    finally:
      monitor.stop()

    if return_code != 0:
      raise RuntimeError("EchoMimicV3-Flash failed:\n" + "\n".join(log_tail))
    if not output_path.is_file():
      raise FileNotFoundError("EchoMimicV3-Flash did not create its output video")

    probe = subprocess.run(
      [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(output_path),
      ],
      check=True,
      capture_output=True,
      text=True,
    )
    gpu_name = subprocess.run(
      ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
      check=True,
      capture_output=True,
      text=True,
    ).stdout.strip()
    return {
      "video": output_path.read_bytes(),
      "elapsed_seconds": round(time.monotonic() - started_at, 2),
      "gpu": gpu_name,
      "peak_gpu_memory_mib": monitor.peak_mib,
      "output_duration_seconds": round(float(probe.stdout.strip()), 3),
      "sample_steps": 8,
      "resolution_limit": "768x768",
    }


@app.local_entrypoint()
def main(
  image: str,
  audio: str,
  output: str,
  prompt: str = (
    "A young man wearing wireless earbuds and a tan jacket speaks naturally "
    "to the camera indoors. Static camera, stable identity, clothing, and "
    "background, subtle conversational gestures."
  ),
) -> None:
  """Upload the shared benchmark inputs and save the generated MP4."""
  image_path = Path(image).expanduser().resolve()
  audio_path = Path(audio).expanduser().resolve()
  output_path = Path(output).expanduser().resolve()
  if not image_path.is_file():
    raise FileNotFoundError(f"Image does not exist: {image_path}")
  if not audio_path.is_file():
    raise FileNotFoundError(f"Audio does not exist: {audio_path}")

  weight_summary = prepare_weights.remote()
  generation_summary = generate_video.remote(
    image_path.read_bytes(),
    audio_path.read_bytes(),
    prompt,
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_bytes(generation_summary.pop("video"))
  print(
    json.dumps(
      {
        "output": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "weights": weight_summary,
        "generation": generation_summary,
      },
      indent=2,
    )
  )
