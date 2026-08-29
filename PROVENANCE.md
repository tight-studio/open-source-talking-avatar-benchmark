# Publication provenance

This repository was assembled from Tight Studio's private application repository on August 29, 2026. The source benchmark article, six model outputs, posters, Modal runners, and measurements were originally published together in commit `388e09d375da29faf45d6ee1f01aeb646a95f7b8` on August 23, 2026.

The standalone publication makes two intentional changes:

1. **Audio removal:** Public result MP4s contain only the original H.264 video track. The AAC narration was removed without re-encoding the video to avoid redistributing a macOS System Voice. SHA-256 values in this repository apply to those public, muted artifacts.
2. **LongCat GPU declaration:** The source runner declared `gpu="H100"` while the recorded benchmark result identified the allocated/tested GPU as H200. The standalone runner requests H200 so its infrastructure declaration matches the published measurement. No timing or memory value was changed.

The repository does not contain the original Modal job logs, cached checkpoints, or source narration waveform. Measurements in `data/results.csv` and `data/results.json` are normalized transcriptions of the recorded benchmark summaries. Output duration and track inventory were rechecked during publication.
