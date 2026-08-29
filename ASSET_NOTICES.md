# Asset and upstream notices

The repository's MIT License applies to original Tight Studio code, report text, and tabular data. It does not relicense third-party source material, upstream projects, model weights, or rights that may exist in generated media.

## Reference portrait

`assets/source/reference.jpg` is a frame from the silent, AI-generated Pexels video [A Man Talking to the Camera](https://www.pexels.com/video/a-man-talking-to-the-camera-8136210/) by AI25.Studio. It is included under the [Pexels license](https://www.pexels.com/license/). Attribution is provided voluntarily. The creator does not endorse Tight Studio or this benchmark.

The poster images are frames from the generated result videos. The result videos visually derive from the same Pexels source frame.

## Audio publication boundary

The measured runs used the sentence:

> Open source avatar models can now generate convincing videos from one image.

The internal benchmark audio was synthesized with the Daniel system voice included with macOS. Apple's macOS Software License Agreement restricts public redistribution of System Voice output. Therefore:

- no source narration file is included;
- all six public result MP4s have had their audio track removed without re-encoding the video track; and
- reruns intended for publication should use speech the operator owns or is licensed to distribute.

The report retains audio-correlation measurements from the original internal runs as numerical observations. Those values do not require redistribution of the source waveform.

## Upstream projects and checkpoints

The runners clone official repositories at pinned commits and download weights from official Hugging Face repositories at pinned revisions. This repository does not include those sources or weights. Review both the code license and checkpoint/model-card terms before running or redistributing outputs.

| Model | Official project | Code license at publication |
|---|---|---|
| Wan2.2-S2V-14B | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) | [Apache-2.0](https://github.com/Wan-Video/Wan2.2/blob/main/LICENSE.txt) |
| LTX-2.3 DubIt | [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) | [LTX-2 Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2) for the tested version |
| LongCat-Video-Avatar 1.5 | [meituan-longcat/LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) | [MIT](https://github.com/meituan-longcat/LongCat-Video/blob/main/LICENSE) |
| LiveAvatar | [Alibaba-Quark/LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar) | [Apache-2.0](https://github.com/Alibaba-Quark/LiveAvatar/blob/main/LICENSE) |
| SoulX-FlashTalk | [Soul-AILab/SoulX-FlashTalk](https://github.com/Soul-AILab/SoulX-FlashTalk) | [Apache-2.0](https://github.com/Soul-AILab/SoulX-FlashTalk/blob/main/LICENSE) |
| EchoMimicV3-Flash | [antgroup/echomimic_v3](https://github.com/antgroup/echomimic_v3) | [Apache-2.0](https://github.com/antgroup/echomimic_v3/blob/main/LICENSE.txt) |
| SoulX-FlashHead | [Soul-AILab/SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) | [Apache-2.0](https://github.com/Soul-AILab/SoulX-FlashHead/blob/main/LICENSE) |

The table summarizes repository code licenses, not necessarily every checkpoint, dependency, training dataset, or output-use term. Licenses can change; the pinned revision and the terms supplied with the downloaded artifact are authoritative.
