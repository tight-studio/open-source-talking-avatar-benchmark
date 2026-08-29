"""Run official LongCat-Video-Avatar 1.5 image-to-video on Modal."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import modal


APP_NAME = "tightstudio-longcat-avatar-15"
CACHE_PATH = Path("/cache")
WEIGHTS_PATH = CACHE_PATH / "weights"
BASE_CHECKPOINT_PATH = WEIGHTS_PATH / "LongCat-Video"
AVATAR_CHECKPOINT_PATH = WEIGHTS_PATH / "LongCat-Video-Avatar-1.5"
REPOSITORY_PATH = Path("/opt/LongCat-Video")

LONGCAT_COMMIT = "6b3f4b8582a8bc3f20f795735f5383716c4ba794"
BASE_REVISION = "03b55529b1d1d4045f5fbe14d65c8c6e8116b278"
AVATAR_REVISION = "92016c71d5d318d0f5d84e4db30015a571484ab6"

BASE_FILES = [
  "tokenizer/*",
  "text_encoder/*",
  "vae/*",
]
AVATAR_FILES = [
  "base_model_int8/*",
  "lora/dmd_lora.safetensors",
  "scheduler/*",
  "vocal_separator/*",
  "whisper-large-v3/*.json",
  "whisper-large-v3/*.txt",
  "whisper-large-v3/model.safetensors",
]

cache_volume = modal.Volume.from_name(
  "tightstudio-longcat-avatar-15-cache",
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
      "MAX_JOBS": "4",
      "TORCH_CUDA_ARCH_LIST": "9.0",
    }
  )
  .run_commands(
    "python -m pip install --upgrade pip wheel setuptools",
    "python -m pip install torch==2.6.0 torchvision==0.21.0 "
    "torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124",
    f"git clone https://github.com/meituan-longcat/LongCat-Video.git "
    f"{REPOSITORY_PATH}",
    f"git -C {REPOSITORY_PATH} checkout {LONGCAT_COMMIT}",
    f"sed '/^torch==/d; /^flash-attn==/d' "
    f"{REPOSITORY_PATH}/requirements.txt > /tmp/requirements-longcat.txt",
    f"sed '/^libsndfile1==/d; /^tritonserverclient==/d' "
    f"{REPOSITORY_PATH}/requirements_avatar.txt "
    f"> /tmp/requirements-longcat-avatar.txt",
    "python -m pip install -r /tmp/requirements-longcat.txt",
    "python -m pip install -r /tmp/requirements-longcat-avatar.txt",
    "python -m pip install flash_attn==2.7.4.post1 --no-build-isolation",
  )
)

app = modal.App(APP_NAME, image=image)


class GpuMemoryMonitor:
  """Poll total GPU memory use while the official subprocess runs."""

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
  """Cache only the official INT8 distilled single-avatar dependencies."""
  from huggingface_hub import snapshot_download

  started_at = time.monotonic()
  snapshot_download(
    repo_id="meituan-longcat/LongCat-Video",
    revision=BASE_REVISION,
    local_dir=BASE_CHECKPOINT_PATH,
    allow_patterns=BASE_FILES,
  )
  snapshot_download(
    repo_id="meituan-longcat/LongCat-Video-Avatar-1.5",
    revision=AVATAR_REVISION,
    local_dir=AVATAR_CHECKPOINT_PATH,
    allow_patterns=AVATAR_FILES,
  )
  cache_volume.commit()

  return {
    "elapsed_seconds": round(time.monotonic() - started_at, 2),
    "cache_bytes": directory_size(CACHE_PATH),
    "base_checkpoint_bytes": directory_size(BASE_CHECKPOINT_PATH),
    "avatar_checkpoint_bytes": directory_size(AVATAR_CHECKPOINT_PATH),
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
  resolution: str = "480p",
) -> dict[str, Any]:
  """Generate an audio-driven video from one reference image."""
  if not image_bytes:
    raise ValueError("image cannot be empty")
  if not audio_bytes:
    raise ValueError("audio cannot be empty")
  if not prompt.strip():
    raise ValueError("prompt cannot be empty")
  if resolution not in {"480p", "720p"}:
    raise ValueError("resolution must be 480p or 720p")

  started_at = time.monotonic()
  with tempfile.TemporaryDirectory(prefix="longcat-avatar-15-") as directory:
    work_path = Path(directory)
    image_path = work_path / "reference.png"
    audio_path = work_path / "speech.wav"
    input_path = work_path / "input.json"
    output_path = work_path / "output"
    image_path.write_bytes(image_bytes)
    audio_path.write_bytes(audio_bytes)

    duration_result = subprocess.run(
      [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
      ],
      check=True,
      capture_output=True,
      text=True,
    )
    audio_duration = float(duration_result.stdout.strip())
    if audio_duration > 10:
      raise ValueError("This trial runner accepts audio up to 10 seconds")

    first_segment_seconds = 93 / 25
    continued_segment_seconds = (93 - 13) / 25
    num_segments = max(
      1,
      1 + math.ceil(
        max(0.0, audio_duration - first_segment_seconds)
        / continued_segment_seconds
      ),
    )
    input_path.write_text(
      json.dumps(
        {
          "prompt": prompt,
          "cond_image": str(image_path),
          "cond_audio": {"person1": str(audio_path)},
        }
      )
    )

    command = [
      "torchrun",
      "--standalone",
      "--nproc_per_node=1",
      str(REPOSITORY_PATH / "run_demo_avatar_single_audio_to_video.py"),
      "--context_parallel_size=1",
      "--checkpoint_dir",
      str(AVATAR_CHECKPOINT_PATH),
      "--stage_1=ai2v",
      "--input_json",
      str(input_path),
      "--output_dir",
      str(output_path),
      "--resolution",
      resolution,
      "--num_segments",
      str(num_segments),
      "--use_distill",
      "--model_type=avatar-v1.5",
      "--use_int8",
    ]

    monitor = GpuMemoryMonitor()
    output_tail: deque[str] = deque(maxlen=160)
    monitor.start()
    try:
      process = subprocess.Popen(
        command,
        cwd=work_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
      )
      assert process.stdout is not None
      for line in process.stdout:
        print(line, end="", flush=True)
        output_tail.append(line.rstrip())
      return_code = process.wait()
    finally:
      monitor.stop()

    if return_code != 0:
      raise RuntimeError(
        "LongCat Avatar 1.5 failed:\n" + "\n".join(output_tail)
      )

    expected_name = (
      f"video_continue_{num_segments}.mp4"
      if num_segments > 1
      else "ai2v_demo_1.mp4"
    )
    result_path = output_path / expected_name
    if not result_path.is_file():
      candidates = sorted(output_path.glob("*.mp4"))
      if not candidates:
        raise FileNotFoundError("LongCat did not create an MP4 output")
      result_path = candidates[-1]

    gpu_result = subprocess.run(
      [
        "nvidia-smi",
        "--query-gpu=name",
        "--format=csv,noheader",
      ],
      check=True,
      capture_output=True,
      text=True,
    )
    return {
      "video": result_path.read_bytes(),
      "elapsed_seconds": round(time.monotonic() - started_at, 2),
      "gpu": gpu_result.stdout.strip(),
      "peak_gpu_memory_mib": monitor.peak_mib,
      "audio_duration_seconds": round(audio_duration, 3),
      "num_segments": num_segments,
      "resolution": resolution,
      "steps": 8,
      "quantization": "INT8 weight-only",
    }


@app.local_entrypoint()
def main(
  image: str,
  audio: str,
  output: str,
  prompt: str = (
    "Static camera. A man wearing glasses and a dark green jacket sits in a "
    "bright, modern home office and speaks naturally to the camera. He makes "
    "subtle head movements and relaxed conversational expressions. Preserve "
    "his identity, clothing, eyeglasses, and the room background."
  ),
  resolution: str = "480p",
) -> None:
  """Upload a reference frame and speech, then save the generated MP4."""
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
    resolution=resolution,
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
