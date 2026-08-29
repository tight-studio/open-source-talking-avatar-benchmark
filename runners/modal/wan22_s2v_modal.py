"""Benchmark the official Wan2.2-S2V-14B pipeline on Modal."""

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


APP_NAME = "tightstudio-wan22-s2v"
CACHE_PATH = Path("/cache")
CHECKPOINT_PATH = CACHE_PATH / "Wan2.2-S2V-14B"
REPOSITORY_PATH = Path("/opt/Wan2.2")
REPOSITORY_COMMIT = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
CHECKPOINT_REVISION = "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b"

CHECKPOINT_FILES = [
  "Wan2.1_VAE.pth",
  "config.json",
  "configuration.json",
  "diffusion_pytorch_model-*.safetensors",
  "diffusion_pytorch_model.safetensors.index.json",
  "google/umt5-xxl/*",
  "models_t5_umt5-xxl-enc-bf16.pth",
  "wav2vec2-large-xlsr-53-english/*",
]

cache_volume = modal.Volume.from_name(
  "tightstudio-wan22-s2v-cache",
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
      "TORCH_CUDA_ARCH_LIST": "9.0",
    }
  )
  .run_commands(
    "python -m pip install --upgrade pip wheel setuptools",
    "python -m pip install torch==2.6.0 torchvision==0.21.0 "
    "torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124",
    f"git clone https://github.com/Wan-Video/Wan2.2.git {REPOSITORY_PATH}",
    f"git -C {REPOSITORY_PATH} checkout {REPOSITORY_COMMIT}",
    f"sed '/^torch/d; /^torchvision/d; /^torchaudio/d; /^flash_attn/d' "
    f"{REPOSITORY_PATH}/requirements.txt > /tmp/wan-requirements.txt",
    "python -m pip install -r /tmp/wan-requirements.txt",
    "python -m pip install decord==0.6.0 librosa==0.11.0 peft==0.17.1",
    "python -m pip install flash_attn==2.7.4.post1 --no-build-isolation",
  )
)

app = modal.App(APP_NAME, image=image)


class GpuMemoryMonitor:
  """Poll total GPU memory while an official inference process runs."""

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
  memory=65536,
  timeout=3 * 60 * 60,
  volumes={str(CACHE_PATH): cache_volume},
)
def prepare_weights() -> dict[str, Any]:
  """Download only the official files required by S2V inference."""
  from huggingface_hub import snapshot_download

  started_at = time.monotonic()
  snapshot_download(
    repo_id="Wan-AI/Wan2.2-S2V-14B",
    revision=CHECKPOINT_REVISION,
    local_dir=CHECKPOINT_PATH,
    allow_patterns=CHECKPOINT_FILES,
  )
  cache_volume.commit()
  return {
    "elapsed_seconds": round(time.monotonic() - started_at, 2),
    "cache_bytes": directory_size(CACHE_PATH),
    "checkpoint_bytes": directory_size(CHECKPOINT_PATH),
  }


@app.function(
  gpu="H200",
  cpu=16,
  memory=131072,
  timeout=4 * 60 * 60,
  max_containers=1,
  volumes={str(CACHE_PATH): cache_volume},
)
def generate_video(
  image_bytes: bytes,
  audio_bytes: bytes,
  prompt: str,
  seed: int = 42,
) -> dict[str, Any]:
  """Generate one short portrait video using the official Wan S2V runner."""
  started_at = time.monotonic()
  with tempfile.TemporaryDirectory(prefix="wan22-s2v-") as directory:
    work_path = Path(directory)
    image_path = work_path / "reference.jpg"
    audio_path = work_path / "speech.wav"
    output_path = work_path / "output.mp4"
    image_path.write_bytes(image_bytes)
    audio_path.write_bytes(audio_bytes)

    command = [
      "python",
      str(REPOSITORY_PATH / "generate.py"),
      "--task",
      "s2v-14B",
      "--size",
      "1024*704",
      "--ckpt_dir",
      str(CHECKPOINT_PATH),
      "--prompt",
      prompt,
      "--image",
      str(image_path),
      "--audio",
      str(audio_path),
      "--infer_frames",
      "92",
      "--num_clip",
      "1",
      "--offload_model",
      "False",
      "--convert_model_dtype",
      "--base_seed",
      str(seed),
      "--save_file",
      str(output_path),
    ]

    monitor = GpuMemoryMonitor()
    log_tail: deque[str] = deque(maxlen=200)
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
      raise RuntimeError("Wan2.2-S2V failed:\n" + "\n".join(log_tail))
    if not output_path.is_file():
      raise FileNotFoundError("Wan2.2-S2V did not create its output video")

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
      "infer_frames": 92,
      "seed": seed,
    }


@app.local_entrypoint()
def main(
  image: str,
  audio: str,
  output: str,
  prompt: str = (
    "A young man wearing wireless earbuds and a tan jacket speaks naturally "
    "to the camera indoors. Static camera, realistic facial movement, stable "
    "identity, clothing, and background."
  ),
) -> None:
  """Upload benchmark inputs, run inference, and save the resulting MP4."""
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
