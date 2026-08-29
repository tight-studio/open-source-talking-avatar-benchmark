"""Benchmark SoulX-FlashTalk and SoulX-FlashHead on Modal."""

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


APP_NAME = "tightstudio-soulx-avatar"
CACHE_PATH = Path("/cache")
FLASHTALK_PATH = Path("/opt/SoulX-FlashTalk")
FLASHHEAD_PATH = Path("/opt/SoulX-FlashHead")
FLASHTALK_WEIGHTS_PATH = CACHE_PATH / "SoulX-FlashTalk-14B"
FLASHHEAD_WEIGHTS_PATH = CACHE_PATH / "SoulX-FlashHead-1_3B"
WAV2VEC_CHINESE_PATH = CACHE_PATH / "chinese-wav2vec2-base"
WAV2VEC_ENGLISH_PATH = CACHE_PATH / "wav2vec2-base-960h"

FLASHTALK_COMMIT = "2ba2e1e801d59a9f526212a5aae5d374db8f2978"
FLASHHEAD_COMMIT = "9bc03de06bb0de82cd6bc477804512ae06144bf2"
FLASHTALK_REVISION = "f3fda6499a6022862cadd39fc41ae63b64706e1b"
FLASHHEAD_REVISION = "59119b6c681230c3eeee157e224ae1941746711e"
WAV2VEC_CHINESE_REVISION = "3991242c806928916fff4a8c0e4f76acf661b743"
WAV2VEC_ENGLISH_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"

FLASHTALK_FILES = [
  "Wan2.1_VAE.pth",
  "config.json",
  "diffusion_pytorch_model-*.safetensors",
  "diffusion_pytorch_model.safetensors.index.json",
  "google/umt5-xxl/*",
  "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
  "models_t5_umt5-xxl-enc-bf16.pth",
  "xlm-roberta-large/*",
]
FLASHHEAD_FILES = [
  "Model_Lite/*",
  "VAE_LTX/*",
  "config.json",
  "model_index.json",
]
WAV2VEC_FILES = [
  "config.json",
  "model.safetensors",
  "preprocessor_config.json",
]

flashtalk_volume = modal.Volume.from_name(
  "tightstudio-soulx-flashtalk-cache",
  create_if_missing=True,
)
flashhead_volume = modal.Volume.from_name(
  "tightstudio-soulx-flashhead-cache",
  create_if_missing=True,
)

image = (
  modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04",
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
    "python -m pip install torch==2.7.1 torchvision==0.22.1 "
    "torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128",
    f"git clone https://github.com/Soul-AILab/SoulX-FlashTalk.git {FLASHTALK_PATH}",
    f"git -C {FLASHTALK_PATH} checkout {FLASHTALK_COMMIT}",
    f"git clone https://github.com/Soul-AILab/SoulX-FlashHead.git {FLASHHEAD_PATH}",
    f"git -C {FLASHHEAD_PATH} checkout {FLASHHEAD_COMMIT}",
    f"sed '/^xformers==/d' {FLASHTALK_PATH}/requirements.txt "
    f"> /tmp/flashtalk-requirements.txt",
    f"sed '/^xformers==/d; /^nvidia-nccl-cu12==/d' "
    f"{FLASHHEAD_PATH}/requirements.txt > /tmp/flashhead-requirements.txt",
    "python -m pip install -r /tmp/flashtalk-requirements.txt",
    "python -m pip install -r /tmp/flashhead-requirements.txt",
    "python -m pip install xformers==0.0.31 "
    "--index-url https://download.pytorch.org/whl/cu128",
    "python -m pip install flash_attn==2.8.0.post2 --no-build-isolation",
  )
)

app = modal.App(APP_NAME, image=image)


class GpuMemoryMonitor:
  """Poll total GPU memory while an official subprocess runs."""

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


def run_process(command: list[str], cwd: Path) -> tuple[int, int, list[str]]:
  monitor = GpuMemoryMonitor()
  log_tail: deque[str] = deque(maxlen=200)
  monitor.start()
  try:
    process = subprocess.Popen(
      command,
      cwd=cwd,
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
  return return_code, monitor.peak_mib, list(log_tail)


@app.function(
  cpu=8,
  memory=65536,
  timeout=3 * 60 * 60,
  volumes={str(CACHE_PATH): flashtalk_volume},
)
def prepare_flashtalk_weights() -> dict[str, Any]:
  """Download the official FlashTalk and audio-encoder weights."""
  from huggingface_hub import snapshot_download

  started_at = time.monotonic()
  snapshot_download(
    repo_id="Soul-AILab/SoulX-FlashTalk-14B",
    revision=FLASHTALK_REVISION,
    local_dir=FLASHTALK_WEIGHTS_PATH,
    allow_patterns=FLASHTALK_FILES,
  )
  snapshot_download(
    repo_id="TencentGameMate/chinese-wav2vec2-base",
    revision=WAV2VEC_CHINESE_REVISION,
    local_dir=WAV2VEC_CHINESE_PATH,
    allow_patterns=HEAD_WAV2VEC_FILES,
  )
  flashtalk_volume.commit()
  return {
    "elapsed_seconds": round(time.monotonic() - started_at, 2),
    "cache_bytes": directory_size(CACHE_PATH),
  }


HEAD_WAV2VEC_FILES = [
  "config.json",
  "preprocessor_config.json",
  "pytorch_model.bin",
]


@app.function(
  cpu=8,
  memory=32768,
  timeout=2 * 60 * 60,
  volumes={str(CACHE_PATH): flashhead_volume},
)
def prepare_flashhead_weights() -> dict[str, Any]:
  """Download the official FlashHead Lite and audio-encoder weights."""
  from huggingface_hub import snapshot_download

  started_at = time.monotonic()
  snapshot_download(
    repo_id="Soul-AILab/SoulX-FlashHead-1_3B",
    revision=FLASHHEAD_REVISION,
    local_dir=FLASHHEAD_WEIGHTS_PATH,
    allow_patterns=FLASHHEAD_FILES,
  )
  snapshot_download(
    repo_id="facebook/wav2vec2-base-960h",
    revision=WAV2VEC_ENGLISH_REVISION,
    local_dir=WAV2VEC_ENGLISH_PATH,
    allow_patterns=WAV2VEC_FILES,
  )
  flashhead_volume.commit()
  return {
    "elapsed_seconds": round(time.monotonic() - started_at, 2),
    "cache_bytes": directory_size(CACHE_PATH),
  }


@app.function(
  gpu="H200",
  cpu=16,
  memory=131072,
  timeout=4 * 60 * 60,
  max_containers=1,
  volumes={str(CACHE_PATH): flashtalk_volume},
)
def generate_flashtalk(
  image_bytes: bytes,
  audio_bytes: bytes,
  prompt: str,
) -> dict[str, Any]:
  """Run the official 14B single-GPU FlashTalk path."""
  started_at = time.monotonic()
  with tempfile.TemporaryDirectory(prefix="soulx-flashtalk-") as directory:
    work_path = Path(directory)
    image_path = work_path / "reference.jpg"
    audio_path = work_path / "speech.wav"
    output_path = work_path / "res_output.mp4"
    image_path.write_bytes(image_bytes)
    audio_path.write_bytes(audio_bytes)
    command = [
      "python",
      str(FLASHTALK_PATH / "generate_video.py"),
      "--ckpt_dir",
      str(FLASHTALK_WEIGHTS_PATH),
      "--wav2vec_dir",
      str(WAV2VEC_CHINESE_PATH),
      "--input_prompt",
      prompt,
      "--cond_image",
      str(image_path),
      "--audio_path",
      str(audio_path),
      "--audio_encode_mode",
      "once",
      "--save_file",
      str(output_path),
    ]
    return_code, peak_mib, log_tail = run_process(command, FLASHTALK_PATH)
    if return_code != 0:
      raise RuntimeError("SoulX-FlashTalk failed:\n" + "\n".join(log_tail))
    if not output_path.is_file():
      raise FileNotFoundError("SoulX-FlashTalk did not create its output video")
    return benchmark_result(output_path, started_at, peak_mib, "14B")


@app.function(
  gpu="L40S",
  cpu=12,
  memory=65536,
  timeout=3 * 60 * 60,
  max_containers=1,
  volumes={str(CACHE_PATH): flashhead_volume},
)
def generate_flashhead(
  image_bytes: bytes,
  audio_bytes: bytes,
) -> dict[str, Any]:
  """Run the official consumer-GPU FlashHead Lite path."""
  started_at = time.monotonic()
  with tempfile.TemporaryDirectory(prefix="soulx-flashhead-") as directory:
    work_path = Path(directory)
    image_path = work_path / "reference.jpg"
    audio_path = work_path / "speech.wav"
    output_path = work_path / "output.mp4"
    image_path.write_bytes(image_bytes)
    audio_path.write_bytes(audio_bytes)
    command = [
      "python",
      str(FLASHHEAD_PATH / "generate_video.py"),
      "--ckpt_dir",
      str(FLASHHEAD_WEIGHTS_PATH),
      "--wav2vec_dir",
      str(WAV2VEC_ENGLISH_PATH),
      "--model_type",
      "lite",
      "--cond_image",
      str(image_path),
      "--audio_path",
      str(audio_path),
      "--audio_encode_mode",
      "once",
      "--save_file",
      str(output_path),
    ]
    return_code, peak_mib, log_tail = run_process(command, FLASHHEAD_PATH)
    if return_code != 0:
      raise RuntimeError("SoulX-FlashHead failed:\n" + "\n".join(log_tail))
    if not output_path.is_file():
      raise FileNotFoundError("SoulX-FlashHead did not create its output video")
    return benchmark_result(output_path, started_at, peak_mib, "1.3B Lite")


def benchmark_result(
  output_path: Path,
  started_at: float,
  peak_mib: int,
  variant: str,
) -> dict[str, Any]:
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
    "peak_gpu_memory_mib": peak_mib,
    "output_duration_seconds": round(float(probe.stdout.strip()), 3),
    "variant": variant,
  }


@app.local_entrypoint()
def main(
  model: str,
  image: str,
  audio: str,
  output: str,
  prompt: str = (
    "A young man wearing wireless earbuds and a tan jacket speaks naturally "
    "to the camera. Only the foreground character moves and the background "
    "remains stable."
  ),
) -> None:
  """Run either FlashTalk or FlashHead with the shared benchmark inputs."""
  if model not in {"flashtalk", "flashhead"}:
    raise ValueError("model must be flashtalk or flashhead")
  image_path = Path(image).expanduser().resolve()
  audio_path = Path(audio).expanduser().resolve()
  output_path = Path(output).expanduser().resolve()
  if not image_path.is_file():
    raise FileNotFoundError(f"Image does not exist: {image_path}")
  if not audio_path.is_file():
    raise FileNotFoundError(f"Audio does not exist: {audio_path}")

  if model == "flashtalk":
    weight_summary = prepare_flashtalk_weights.remote()
    generation_summary = generate_flashtalk.remote(
      image_path.read_bytes(),
      audio_path.read_bytes(),
      prompt,
    )
  else:
    weight_summary = prepare_flashhead_weights.remote()
    generation_summary = generate_flashhead.remote(
      image_path.read_bytes(),
      audio_path.read_bytes(),
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
