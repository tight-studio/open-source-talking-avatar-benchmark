# Reproducing the benchmark

## Scope

The repository captures the official runnable paths used for the August 23, 2026 benchmark. The runners pin upstream Git commits and Hugging Face revisions, build model-specific containers, cache weights in Modal Volumes, execute inference, poll `nvidia-smi` every 0.5 seconds, and print a JSON summary.

Reruns can incur substantial GPU, CPU, memory, storage, and network charges. The reported cost excludes the initial internet download of model weights but includes model loading, preprocessing, generation, decoding, and muxing inside the measured generation function.

## Prerequisites

1. Python 3.10 or newer.
2. A configured [Modal](https://modal.com/) account with permission to use the requested GPUs.
3. Access to every official model repository and checkpoint you plan to run.
4. A portrait and speech file you have permission to process and publish.

Install and authenticate:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m modal setup
```

## Shared input protocol

The measured image-to-video runs used `assets/source/reference.jpg` and the same 3.63-second spoken sentence. The original system-voice audio is not redistributed; supply a rights-cleared WAV when reproducing. Record its duration, sample rate, voice source, and checksum with the new run.

The runners use model-specific prompts because their official interfaces and conditioning conventions differ. Defaults are preserved in each file. Do not describe a rerun as exact if you change the prompt, seed, resolution, steps, quantization, checkpoint, or GPU class.

## Commands

Replace `<speech.wav>`, `<reference-video.mp4>`, and output paths with your files.

```bash
modal run runners/modal/wan22_s2v_modal.py \
  --image assets/source/reference.jpg \
  --audio <speech.wav> \
  --output tmp/wan22.mp4

modal run runners/modal/longcat_avatar_15_modal.py \
  --image assets/source/reference.jpg \
  --audio <speech.wav> \
  --output tmp/longcat.mp4

modal run runners/modal/liveavatar_modal.py \
  --image assets/source/reference.jpg \
  --audio <speech.wav> \
  --output tmp/liveavatar.mp4

modal run runners/modal/soulx_avatar_modal.py \
  --model flashtalk \
  --image assets/source/reference.jpg \
  --audio <speech.wav> \
  --output tmp/flashtalk.mp4

modal run runners/modal/soulx_avatar_modal.py \
  --model flashhead \
  --image assets/source/reference.jpg \
  --audio <speech.wav> \
  --output tmp/flashhead.mp4

modal run runners/modal/echomimic_v3_flash_modal.py \
  --image assets/source/reference.jpg \
  --audio <speech.wav> \
  --output tmp/echomimic.mp4

modal run runners/modal/ltx23_dubit_modal.py \
  --reference-video <reference-video.mp4> \
  --output tmp/ltx23-dubit.mp4
```

LTX-2.3 DubIt requires its own voiced reference video and official checkpoint authorization. The benchmark stopped during weight preparation because that authorization was unavailable; no substitute mirror was used.

## Measurement definitions

- **Elapsed time:** Wall-clock time inside the generation function from container start until the final MP4 is ready to return. Persistent weight download time is excluded; model load is included.
- **Peak VRAM:** Maximum total `memory.used` reported by `nvidia-smi` while the official inference subprocess ran, polled every 0.5 seconds.
- **Model cache:** Size of the persistent cache after the files selected by the runner were present.
- **Estimated compute:** Modal list-price estimate for the requested GPU, CPU, and memory over the measured function duration. Storage, initial downloads, and internet transfer are excluded.
- **Always-on GPU/month:** GPU-only capacity estimate for 720 hours. Modal can scale to zero, so this is not a minimum bill.
- **Audio correlation:** Normalized waveform correlation after extracting and aligning the output audio with the supplied input. Small differences reflect resampling and AAC encoding.

## Publication checklist

Before publishing a rerun:

1. Save the runner commit and its upstream/checkpoint revisions.
2. Save the complete JSON summary and job logs.
3. Verify output duration, dimensions, frame rate, and track inventory.
4. Record SHA-256 checksums for inputs and outputs.
5. Confirm portrait, voice, checkpoint, and generated-output rights.
6. Remove or replace speech that cannot be publicly redistributed.
7. Describe failures as failures; do not silently substitute unofficial weights.
