"""Benchmark the official LTX-2.3 DubIt pipeline on Modal."""

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


APP_NAME = "tightstudio-ltx23-dubit"
CACHE_PATH = Path("/cache")
MODEL_PATH = CACHE_PATH / "LTX-2.3"
GEMMA_PATH = CACHE_PATH / "gemma-3-12b-it-qat-q4_0-unquantized"
LORA_PATH = CACHE_PATH / "LTX-2.3-22b-IC-LoRA-DubIt"
REPOSITORY_PATH = Path("/opt/ltx2")
REPOSITORY_COMMIT = "400fd31054597515f47125691032c04b1c3ee24e"
MODEL_REVISION = "6b5a83e3045eaf8e46cfa0acce512412aa2b9cce"
GEMMA_REVISION = "d62fe4f1995ade703b49a0f3c0d0f161237ef437"
LORA_REVISION = "1456334c3d69924de5083e553733b108ed1147f2"

DISTILLED_FILENAME = "ltx-2.3-22b-distilled-1.1.safetensors"
UPSCALER_FILENAME = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
LORA_FILENAME = "ltx-2.3-22b-ic-lora-dubit-0.9.safetensors"

cache_volume = modal.Volume.from_name(
  "tightstudio-ltx23-dubit-cache",
  create_if_missing=True,
)

image = (
  modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04",
    add_python="3.11",
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
      "TOKENIZERS_PARALLELISM": "false",
    }
  )
  .run_commands(
    "python -m pip install --upgrade pip wheel setuptools",
    "python -m pip install torch==2.9.1 torchvision==0.24.1 "
    "torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128",
    f"git clone https://github.com/Lightricks/LTX-2.git {REPOSITORY_PATH}",
    f"git -C {REPOSITORY_PATH} checkout {REPOSITORY_COMMIT}",
    f"python -m pip install -e {REPOSITORY_PATH}/packages/ltx-core "
    f"-e {REPOSITORY_PATH}/packages/ltx-pipelines huggingface_hub",
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
  timeout=4 * 60 * 60,
  volumes={str(CACHE_PATH): cache_volume},
)
def prepare_weights() -> dict[str, Any]:
  """Cache the LTX-2.3 monolith, upscaler, Gemma, and DubIt LoRA."""
  from huggingface_hub import hf_hub_download, snapshot_download

  started_at = time.monotonic()
  MODEL_PATH.mkdir(parents=True, exist_ok=True)
  LORA_PATH.mkdir(parents=True, exist_ok=True)
  hf_hub_download(
    repo_id="Lightricks/LTX-2.3",
    filename=DISTILLED_FILENAME,
    revision=MODEL_REVISION,
    local_dir=MODEL_PATH,
  )
  hf_hub_download(
    repo_id="Lightricks/LTX-2.3",
    filename=UPSCALER_FILENAME,
    revision=MODEL_REVISION,
    local_dir=MODEL_PATH,
  )
  hf_hub_download(
    repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-DubIt",
    filename=LORA_FILENAME,
    revision=LORA_REVISION,
    local_dir=LORA_PATH,
  )
  snapshot_download(
    repo_id="Lightricks/gemma-3-12b-it-qat-q4_0-unquantized",
    revision=GEMMA_REVISION,
    local_dir=GEMMA_PATH,
  )
  cache_volume.commit()
  return {
    "elapsed_seconds": round(time.monotonic() - started_at, 2),
    "cache_bytes": directory_size(CACHE_PATH),
    "model_bytes": directory_size(MODEL_PATH),
    "gemma_bytes": directory_size(GEMMA_PATH),
    "lora_bytes": directory_size(LORA_PATH),
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
  reference_video_bytes: bytes,
  prompt: str,
) -> dict[str, Any]:
  """Generate one DubIt rephrasing clip with the distilled LTX-2.3 model."""
  started_at = time.monotonic()
  with tempfile.TemporaryDirectory(prefix="ltx23-dubit-") as directory:
    work_path = Path(directory)
    reference_path = work_path / "reference.mp4"
    output_path = work_path / "output.mp4"
    reference_path.write_bytes(reference_video_bytes)

    command = [
      "python",
      "-m",
      "ltx_pipelines.dubit",
      "--distilled-checkpoint-path",
      str(MODEL_PATH / DISTILLED_FILENAME),
      "--gemma-root",
      str(GEMMA_PATH),
      "--spatial-upsampler-path",
      str(MODEL_PATH / UPSCALER_FILENAME),
      "--reference-video",
      str(reference_path),
      "--reference-strength",
      "1.0",
      "--lora",
      str(LORA_PATH / LORA_FILENAME),
      "1.0",
      "--prompt",
      prompt,
      "--height",
      "768",
      "--width",
      "448",
      "--seed",
      "42",
      "--quantization",
      "fp8-cast",
      "--output-path",
      str(output_path),
    ]

    monitor = GpuMemoryMonitor()
    log_tail: deque[str] = deque(maxlen=240)
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
      raise RuntimeError("LTX-2.3 DubIt failed:\n" + "\n".join(log_tail))
    if not output_path.is_file():
      raise FileNotFoundError("LTX-2.3 DubIt did not create its output video")

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
      "quantization": "fp8-cast",
      "resolution": "448x768",
      "reference_strength": 1.0,
    }


@app.local_entrypoint()
def main(
  reference_video: str,
  output: str,
  prompt: str = (
    "A young man wearing wireless earbuds and a tan jacket speaks naturally "
    "to the camera in a calm male voice, saying, 'Open source avatar models "
    "can now generate convincing videos from one image.' Static camera, "
    "stable identity, clothing, and background."
  ),
) -> None:
  """Upload the voiced reference clip and save the generated MP4."""
  reference_path = Path(reference_video).expanduser().resolve()
  output_path = Path(output).expanduser().resolve()
  if not reference_path.is_file():
    raise FileNotFoundError(f"Reference video does not exist: {reference_path}")

  weight_summary = prepare_weights.remote()
  generation_summary = generate_video.remote(
    reference_path.read_bytes(),
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
