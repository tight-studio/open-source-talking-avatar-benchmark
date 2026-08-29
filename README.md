# Open-source talking-avatar benchmark (2026)

[![Technical report](https://img.shields.io/badge/report-PDF-E5484D)](report/open-source-talking-avatar-benchmark-2026.pdf)
[![Validate](https://github.com/tight-studio/open-source-talking-avatar-benchmark/actions/workflows/validate.yml/badge.svg)](https://github.com/tight-studio/open-source-talking-avatar-benchmark/actions/workflows/validate.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-0F766E)](LICENSE)

We ran every talking-avatar model that met two predeclared criteria on August 23, 2026: an official usable release during the preceding year and more than 1,000 GitHub stars. Six of seven qualifying models completed from the same public portrait and 3.63-second narration.

This repository publishes the runnable Modal harnesses, machine-readable measurements, muted result videos, checksums, and the full [technical report](REPORT.md). It accompanies Tight Studio's [plain-language benchmark article](https://tight.studio/blog/open-source-talking-avatar-models/).

> This is a deployment benchmark, not a controlled perceptual study. We intentionally used each project's recommended runnable path, so resolution, inference steps, quantization, and GPU class differ.

## Main findings

| Finding | Result |
|---|---|
| Sharpest heavyweight result | **LiveAvatar**, 181.44 s on H200, 61,401 MiB peak VRAM |
| Best deployment balance | **EchoMimicV3-Flash**, 251.94 s on L40S, 33,726 MiB peak VRAM |
| Most expressive motion | **LongCat-Video-Avatar 1.5**, 233.08 s on H200 |
| Lowest memory | **SoulX-FlashHead Lite**, 5,763 MiB peak VRAM |
| Slowest successful run | **Wan2.2-S2V-14B**, 780.29 s for a 3.813 s output |
| Blocked run | **LTX-2.3 DubIt**, official checkpoint approval was unavailable |

## Result gallery

The public MP4s are intentionally muted. GitHub renders the attached H.264 videos below as inline players.

### Wan2.2-S2V-14B

https://github.com/user-attachments/assets/14bf6509-e050-4820-9683-bfabbe9acb7f

Stable identity and background with restrained motion; 512 x 896 at 16 fps.

### LongCat-Video-Avatar 1.5

https://github.com/user-attachments/assets/9763029d-2bee-43f3-be62-eda3a43db6e5

The most expressive head and body performance, including motion absent from the reference frame; 480 x 832 at 25 fps.

### LiveAvatar

https://github.com/user-attachments/assets/6a3dc009-3e6d-4285-a38c-c2204893ca82

The sharpest heavyweight identity with convincing, restrained expression; 384 x 704 at 25 fps.

### SoulX-FlashTalk

https://github.com/user-attachments/assets/41fde8ef-a1ed-4a3f-8048-9bfb25b243bb

A stable face with clear mouth motion; 416 x 720 at 25 fps.

### EchoMimicV3-Flash

https://github.com/user-attachments/assets/47b76f60-cf70-4606-99ce-568b177c6128

The strongest quality-to-infrastructure balance in this test; 480 x 848 at 25 fps.

### SoulX-FlashHead Lite

https://github.com/user-attachments/assets/aa3502bf-99ae-4db7-a0f8-c58eaa1ad608

The lowest-memory result, with a tighter crop and softer identity; 512 x 512 at 25 fps.

## Repository map

```text
assets/           Reference image and result posters
data/             CSV and JSON benchmark records
report/           Versioned PDF technical report
results/videos/   Muted generated outputs
runners/modal/    Pinned Modal runners for all seven models
scripts/          Artifact validation and audio-removal utilities
REPORT.md         Technical report source
REPRODUCING.md    Setup and rerun commands
```

## Reproduce or inspect

The quickest no-GPU check is:

```bash
python3 scripts/validate_repository.py
```

Rerunning inference requires a Modal account, access to the relevant weights, and paid GPU compute. See [REPRODUCING.md](REPRODUCING.md) before starting a job.

## Responsible use

Use only portraits, videos, and voices you own or have permission to animate. Do not use these runners to impersonate people, bypass consent, or create deceptive media. The included portrait came from an AI-generated Pexels video; its creator does not endorse this work.

## Licensing and media notices

Original Tight Studio code, text, and tabular data are released under the [MIT License](LICENSE). The reference frame, generated videos, model repositories, and checkpoints have separate terms described in [ASSET_NOTICES.md](ASSET_NOTICES.md). No model weights or upstream model source code are redistributed here.

## Citation

Citation metadata is available in [CITATION.cff](CITATION.cff). In prose:

> Jiang, Ethan, and Tight Studio. 2026. *A Deployment Benchmark of Popular Open-Source Talking-Avatar Models Released in 2025-2026*. Version 1.0.0.
