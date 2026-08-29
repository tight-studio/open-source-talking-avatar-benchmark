"""Benchmark the official LiveAvatar single-GPU pipeline on Modal."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import modal


APP_NAME = "tightstudio-liveavatar"
CACHE_PATH = Path("/cache")
WAN_PATH = CACHE_PATH / "Wan2.2-S2V-14B"
LORA_PATH = CACHE_PATH / "Live-Avatar" / "liveavatar.safetensors"
REPOSITORY_PATH = Path("/opt/LiveAvatar")
REPOSITORY_COMMIT = "9c2c38de1715e553127b42c82cb959f756b02595"
WAN_REVISION = "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b"
LORA_REVISION = "92cdccd12a91e8a63767a7c821b7c75e51d5a172"

WAN_FILES = [
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
    "nvidia/cuda:12.8.1-devel-ubuntu22.04",
    add_python="3.10",
  )
  .apt_install(
    "build-essential",
    "clang",
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
    "python -m pip install torch==2.8.0 torchvision==0.23.0 "
    "torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128",
    f"git clone https://github.com/Alibaba-Quark/LiveAvatar.git {REPOSITORY_PATH}",
    f"git -C {REPOSITORY_PATH} checkout {REPOSITORY_COMMIT}",
    f"sed '/^torch/d; /^torchvision/d; /^torchaudio/d' "
    f"{REPOSITORY_PATH}/requirements.txt > /tmp/liveavatar-requirements.txt",
    "python -m pip install -r /tmp/liveavatar-requirements.txt",
    "python -m pip install flash_attn==2.8.3 --no-build-isolation",
    "python -m pip uninstall -y deepspeed",
  )
)

app = modal.App(APP_NAME, image=image)


class GpuMemoryMonitor:
  """Poll total GPU memory while the official runner executes."""

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
  """Cache the shared Wan base and LiveAvatar LoRA."""
  from huggingface_hub import hf_hub_download, snapshot_download

  started_at = time.monotonic()
  snapshot_download(
    repo_id="Wan-AI/Wan2.2-S2V-14B",
    revision=WAN_REVISION,
    local_dir=WAN_PATH,
    allow_patterns=WAN_FILES,
  )
  hf_hub_download(
    repo_id="Quark-Vision/Live-Avatar",
    filename="liveavatar.safetensors",
    revision=LORA_REVISION,
    local_dir=LORA_PATH.parent,
  )
  cache_volume.commit()
  return {
    "elapsed_seconds": round(time.monotonic() - started_at, 2),
    "cache_bytes": directory_size(CACHE_PATH),
    "wan_bytes": directory_size(WAN_PATH),
    "liveavatar_bytes": directory_size(LORA_PATH.parent),
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
) -> dict[str, Any]:
  """Run the official four-step FP8 single-GPU LiveAvatar path."""
  started_at = time.monotonic()
  with tempfile.TemporaryDirectory(prefix="liveavatar-") as directory:
    work_path = Path(directory)
    image_path = work_path / "reference.jpg"
    audio_path = work_path / "speech.wav"
    output_path = work_path / "output.mp4"
    image_path.write_bytes(image_bytes)
    audio_path.write_bytes(audio_bytes)

    command = [
      "python",
      str(REPOSITORY_PATH / "minimal_inference/s2v_streaming_interact.py"),
      "--ulysses_size",
      "1",
      "--task",
      "s2v-14B",
      "--size",
      "704*384",
      "--base_seed",
      "420",
      "--training_config",
      str(REPOSITORY_PATH / "liveavatar/configs/s2v_causal_sft.yaml"),
      "--offload_model",
      "True",
      "--convert_model_dtype",
      "--prompt",
      prompt,
      "--image",
      str(image_path),
      "--audio",
      str(audio_path),
      "--infer_frames",
      "48",
      "--load_lora",
      "--lora_path_dmd",
      str(LORA_PATH),
      "--sample_steps",
      "4",
      "--sample_guide_scale",
      "0",
      "--num_clip",
      "2",
      "--num_gpus_dit",
      "1",
      "--sample_solver",
      "euler",
      "--single_gpu",
      "--ckpt_dir",
      str(WAN_PATH),
      "--fp8",
      "--save_file",
      str(output_path),
    ]

    monitor = GpuMemoryMonitor()
    log_tail: deque[str] = deque(maxlen=220)
    monitor.start()
    try:
      process = subprocess.Popen(
        command,
        cwd=REPOSITORY_PATH,
        env={
          **os.environ,
          "MASTER_ADDR": "127.0.0.1",
          "MASTER_PORT": "29500",
          "RANK": "0",
          "LOCAL_RANK": "0",
          "WORLD_SIZE": "1",
          "ENABLE_COMPILE": "false",
        },
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
      raise RuntimeError("LiveAvatar failed:\n" + "\n".join(log_tail))
    if not output_path.is_file():
      raise FileNotFoundError("LiveAvatar did not create its output video")

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
      "sample_steps": 4,
      "quantization": "FP8",
    }


@app.local_entrypoint()
def main(
  image: str,
  audio: str,
  output: str,
  prompt: str = (
    "A young man wearing wireless earbuds and a tan jacket speaks naturally "
    "to the camera indoors. Static camera, realistic expressions, stable "
    "identity, clothing, and background."
  ),
) -> None:
  """Upload benchmark inputs, run LiveAvatar, and save the MP4."""
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
