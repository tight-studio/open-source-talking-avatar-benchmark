#!/usr/bin/env python3
"""Build the versioned PDF technical report from repository data and assets."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
  Flowable,
  Image,
  KeepTogether,
  PageBreak,
  Paragraph,
  SimpleDocTemplate,
  Spacer,
  Table,
  TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report/open-source-talking-avatar-benchmark-2026.pdf"
DATA = json.loads((ROOT / "data/results.json").read_text())

INK = colors.HexColor("#101828")
MUTED = colors.HexColor("#475467")
LIGHT = colors.HexColor("#F2F4F7")
TEAL = colors.HexColor("#0F766E")
TEAL_LIGHT = colors.HexColor("#E6F5F2")
CORAL = colors.HexColor("#E5484D")
GOLD = colors.HexColor("#B7791F")
WHITE = colors.white


class MetricBars(Flowable):
  """A compact horizontal-bar comparison chart."""

  def __init__(self, title: str, rows: list[tuple[str, float, str]], width: float):
    super().__init__()
    self.title = title
    self.rows = rows
    self.width = width
    self.height = 34 + 26 * len(rows)

  def draw(self) -> None:
    canvas = self.canv
    canvas.setFont("Helvetica-Bold", 11)
    canvas.setFillColor(INK)
    canvas.drawString(0, self.height - 12, self.title)
    maximum = max(value for _, value, _ in self.rows)
    label_width = 142
    value_width = 56
    bar_width = self.width - label_width - value_width - 10
    y = self.height - 36
    for index, (label, value, display) in enumerate(self.rows):
      canvas.setFont("Helvetica", 8.4)
      canvas.setFillColor(MUTED)
      canvas.drawString(0, y + 3, label)
      canvas.setFillColor(LIGHT)
      canvas.roundRect(label_width, y, bar_width, 10, 3, fill=1, stroke=0)
      fill = bar_width * (value / maximum)
      canvas.setFillColor(TEAL if index != len(self.rows) - 1 else CORAL)
      canvas.roundRect(label_width, y, max(4, fill), 10, 3, fill=1, stroke=0)
      canvas.setFont("Helvetica-Bold", 8.4)
      canvas.setFillColor(INK)
      canvas.drawRightString(self.width, y + 3, display)
      y -= 26


def styles() -> dict[str, ParagraphStyle]:
  base = getSampleStyleSheet()
  return {
    "title": ParagraphStyle(
      "Title",
      parent=base["Title"],
      fontName="Helvetica-Bold",
      fontSize=26,
      leading=29,
      textColor=INK,
      alignment=TA_LEFT,
      spaceAfter=14,
    ),
    "subtitle": ParagraphStyle(
      "Subtitle",
      parent=base["Normal"],
      fontName="Helvetica",
      fontSize=11.5,
      leading=16,
      textColor=MUTED,
      spaceAfter=14,
    ),
    "h1": ParagraphStyle(
      "Heading1",
      parent=base["Heading1"],
      fontName="Helvetica-Bold",
      fontSize=17,
      leading=21,
      textColor=INK,
      spaceBefore=8,
      spaceAfter=8,
      keepWithNext=True,
    ),
    "h2": ParagraphStyle(
      "Heading2",
      parent=base["Heading2"],
      fontName="Helvetica-Bold",
      fontSize=12,
      leading=15,
      textColor=TEAL,
      spaceBefore=7,
      spaceAfter=4,
      keepWithNext=True,
    ),
    "body": ParagraphStyle(
      "Body",
      parent=base["BodyText"],
      fontName="Helvetica",
      fontSize=9.2,
      leading=13.3,
      textColor=INK,
      spaceAfter=7,
    ),
    "small": ParagraphStyle(
      "Small",
      parent=base["BodyText"],
      fontName="Helvetica",
      fontSize=7.7,
      leading=10.2,
      textColor=MUTED,
      spaceAfter=4,
    ),
    "callout": ParagraphStyle(
      "Callout",
      parent=base["BodyText"],
      fontName="Helvetica-Bold",
      fontSize=10,
      leading=14,
      textColor=TEAL,
      backColor=TEAL_LIGHT,
      borderColor=TEAL,
      borderWidth=0.7,
      borderPadding=10,
      spaceBefore=15,
      spaceAfter=10,
    ),
    "caption": ParagraphStyle(
      "Caption",
      parent=base["BodyText"],
      fontName="Helvetica",
      fontSize=7.4,
      leading=9.5,
      textColor=MUTED,
      alignment=TA_CENTER,
      spaceBefore=3,
    ),
    "cover_meta": ParagraphStyle(
      "CoverMeta",
      parent=base["BodyText"],
      fontName="Helvetica-Bold",
      fontSize=8.5,
      leading=12,
      textColor=TEAL,
      spaceAfter=5,
    ),
  }


S = styles()


def p(text: str, style: str = "body") -> Paragraph:
  return Paragraph(text, S[style])


def bullet(text: str) -> Paragraph:
  style = ParagraphStyle(
    "Bullet",
    parent=S["body"],
    leftIndent=14,
    firstLineIndent=-8,
    bulletIndent=0,
    spaceAfter=4,
  )
  return Paragraph(text, style, bulletText="•")


def table(data: list[list[object]], widths: list[float], font_size: float = 7.2) -> Table:
  result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
  result.setStyle(
    TableStyle(
      [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D5DD")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
      ]
    )
  )
  return result


def metric_tile(label: str, value: str, note: str) -> Table:
  content = [
    [Paragraph(label.upper(), S["small"])],
    [Paragraph(value, ParagraphStyle("Metric", parent=S["h1"], textColor=TEAL, spaceAfter=1))],
    [Paragraph(note, S["small"])],
  ]
  tile = Table(content, colWidths=[1.55 * inch])
  tile.setStyle(
    TableStyle(
      [
        ("BACKGROUND", (0, 0), (-1, -1), TEAL_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#A7D9D2")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
      ]
    )
  )
  return tile


def footer(canvas, doc) -> None:
  canvas.saveState()
  width, _ = letter
  canvas.setStrokeColor(colors.HexColor("#D0D5DD"))
  canvas.setLineWidth(0.4)
  canvas.line(doc.leftMargin, 0.49 * inch, width - doc.rightMargin, 0.49 * inch)
  canvas.setFont("Helvetica", 7.5)
  canvas.setFillColor(MUTED)
  canvas.drawString(doc.leftMargin, 0.31 * inch, "Tight Studio · Open-source talking-avatar benchmark · v1.0.0")
  canvas.drawRightString(width - doc.rightMargin, 0.31 * inch, str(doc.page))
  canvas.restoreState()


def build_story() -> list[Flowable]:
  completed = [item for item in DATA["results"] if item["status"] == "completed"]
  story: list[Flowable] = []

  story.append(p("TIGHT STUDIO · TECHNICAL REPORT 1.0.0", "cover_meta"))
  story.append(Spacer(1, 0.18 * inch))
  story.append(p("A Deployment Benchmark of Popular Open-Source Talking-Avatar Models Released in 2025-2026", "title"))
  story.append(p("Seven qualifying projects. One public portrait. One short narration. Six completed official pipelines measured for time, GPU memory, storage, cost, and output behavior.", "subtitle"))
  hero = Image(str(ROOT / "assets/source/reference.jpg"), width=3.23 * inch, height=3.23 * inch)
  hero.hAlign = "LEFT"
  cover_summary = Table(
    [
      [hero, p("<b>Benchmark date</b><br/>August 23, 2026<br/><br/><b>Author</b><br/>Ethan Jiang, Tight Studio<br/><br/><b>Publication date</b><br/>August 29, 2026<br/><br/><b>Artifact</b><br/><font size='8'>github.com/tight-studio/<br/>open-source-talking-avatar-benchmark</font>", "body")],
    ],
    colWidths=[3.35 * inch, 2.55 * inch],
  )
  cover_summary.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 12)]))
  story.append(cover_summary)
  story.append(Spacer(1, 0.22 * inch))
  story.append(Table([[metric_tile("Fastest H200", "181.44 s", "LiveAvatar"), metric_tile("Lowest VRAM", "5,763 MiB", "FlashHead Lite"), metric_tile("Best balance", "L40S", "EchoMimicV3-Flash")]], colWidths=[1.82 * inch] * 3, hAlign="LEFT"))
  story.append(Spacer(1, 0.22 * inch))
  story.append(p("<b>Publication note.</b> Public result MP4s are muted because the internal test narration used a macOS System Voice that cannot be publicly redistributed under Apple's license. The H.264 video tracks were copied without re-encoding.", "callout"))
  story.append(PageBreak())

  story.append(p("Abstract", "h1"))
  story.append(p("Talking-avatar projects claim real-time generation, identity preservation, expressive motion, and long-duration stability, but their examples use different inputs, resolutions, hardware, and timing boundaries. We evaluated every project that met two predeclared eligibility rules on August 23, 2026: an official usable release during the prior year and more than 1,000 GitHub stars. Seven projects qualified. Six produced videos; LTX-2.3 DubIt was blocked before GPU allocation because the official checkpoint required approval unavailable to the test environment."))
  story.append(p("Peak VRAM ranged from 5,763 MiB for SoulX-FlashHead Lite to 61,401 MiB for LiveAvatar. End-to-end measured generation time ranged from 181.44 seconds for LiveAvatar to 780.29 seconds for Wan2.2-S2V. Qualitative inspection found LiveAvatar to be the sharpest heavyweight result, EchoMimicV3-Flash the strongest practical balance, LongCat the most expressive, and FlashHead Lite the least memory-intensive. These are deployable official-path observations, not a controlled human-preference ranking."))
  story.append(p("1. Research question", "h1"))
  story.append(p("Which recent, meaningfully adopted open-source talking-avatar projects can turn a single portrait and short narration into a convincing video, and what does each official runnable path require in time, memory, storage, and compute cost? Deployment decisions combine visual behavior with checkpoint access, dependency reliability, memory headroom, compilation, and serving cost."))
  story.append(p("2. Cohort selection", "h1"))
  story.append(bullet("Official usable code and weights first released from August 23, 2025 through August 23, 2026."))
  story.append(bullet("More than 1,000 stars on the official GitHub repository on August 23, 2026."))
  cohort = [["Model", "Stars", "Release", "Input", "Outcome"]]
  inputs = {
    "Wan2.2-S2V-14B": "Image + audio",
    "LTX-2.3 DubIt": "Video + text",
    "LongCat-Video-Avatar 1.5": "Image + audio",
    "LiveAvatar": "Image + audio",
    "SoulX-FlashTalk": "Image + audio",
    "EchoMimicV3-Flash": "Image + audio",
    "SoulX-FlashHead Lite": "Image + audio",
  }
  for item in DATA["results"]:
    cohort.append([item["model"], f'{item["stars_as_of_observation_date"]:,}', item["public_release"], inputs[item["model"]], item["status"].title()])
  story.append(table(cohort, [1.58 * inch, 0.55 * inch, 0.82 * inch, 0.95 * inch, 1.1 * inch], 6.7))
  story.append(p("GitHub stars measure attention, not quality. Projects released before the cutoff were excluded regardless of continued popularity.", "small"))

  story.append(PageBreak())
  story.append(p("3. Protocol and measurement boundary", "h1"))
  story.append(p("Every successful run used the same public reference frame and the same 3.63-second sentence. Runs were deployed on Modal in model-specific CUDA containers. Each runner pins the official Git commit and checkpoint revision. Weights were already present on persistent Modal Volumes before timed generation."))
  story.append(p("The clock begins when the generation container starts and stops when the final MP4 is ready to return. It includes model loading, preprocessing, inference, decoding, and muxing, while excluding one-time internet weight download. Peak VRAM is the maximum total <font name='Courier'>memory.used</font> reported by <font name='Courier'>nvidia-smi</font> during the official subprocess, sampled every 0.5 seconds."))
  story.append(p("Measured fields", "h2"))
  for text in [
    "Generation-function wall time and output duration.",
    "Peak GPU memory, output dimensions, and frame rate.",
    "Persistent cache size and point-in-time Modal compute estimate.",
    "Normalized waveform correlation between supplied and output audio.",
  ]:
    story.append(bullet(text))
  story.append(p("The official pipelines use different resolutions, prompts, inference steps, quantization, and GPU classes. That is part of the deployment result, but it prevents architecture-only speed or controlled perceptual claims.", "callout"))

  story.append(p("4. Quantitative results", "h1"))
  result_rows = [["Model", "GPU", "Peak MiB", "Time", "Output", "Cost", "Cache"]]
  for item in DATA["results"]:
    if item["status"] == "blocked":
      result_rows.append([item["model"], "-", "-", "Blocked", "-", "$0", "Gated"])
      continue
    cache = item["model_cache"]
    result_rows.append([
      item["model"], item["gpu_tested"], f'{item["peak_vram_mib"]:,}', f'{item["elapsed_seconds"]:.2f}s',
      f'{item["output"]["width"]}x{item["output"]["height"]}\n{item["output"]["fps"]} fps',
      f'${item["estimated_compute_usd"]:.4f}', f'{cache["value"]:.2f} {cache["unit"]}',
    ])
  story.append(table(result_rows, [1.32 * inch, 0.43 * inch, 0.65 * inch, 0.62 * inch, 0.72 * inch, 0.62 * inch, 0.75 * inch], 6.2))

  story.append(PageBreak())
  vram_rows = sorted([(item["model"], item["peak_vram_mib"], f'{item["peak_vram_mib"]:,}') for item in completed], key=lambda row: row[1])
  time_rows = sorted([(item["model"], item["elapsed_seconds"], f'{item["elapsed_seconds"]:.2f}s') for item in completed], key=lambda row: row[1])
  story.append(p("Infrastructure comparison", "h1"))
  story.append(MetricBars("Peak VRAM (MiB)", vram_rows, 6.05 * inch))
  story.append(Spacer(1, 0.18 * inch))
  story.append(MetricBars("Measured cold-job time", time_rows, 6.05 * inch))
  story.append(Spacer(1, 0.15 * inch))
  story.append(p("FlashHead Lite is the only measured full generator that clearly falls below 16 GB. EchoMimic fits in a 48 GB class with comfortable headroom. LongCat's 46,827 MiB is operationally tight for a nominal 48 GB card. Wan, LiveAvatar, FlashTalk, and LongCat are safest on 80 GB-class hardware for these configurations."))
  story.append(p("No completed cold job generated faster than playback duration. Compilation-heavy SoulX models improve dramatically on later chunks, so warmed streaming latency should be measured separately from these end-to-end cold jobs.", "callout"))

  story.append(PageBreak())
  story.append(p("5. Visual results", "h1"))
  poster_info = [
    ("Wan2.2-S2V-14B", "wan22-s2v-stock-avatar-poster.jpg", "Stable identity; restrained motion; 16 fps."),
    ("LongCat Avatar 1.5", "longcat-stock-avatar-poster.jpg", "Most expressive motion; occasional mouth exaggeration."),
    ("LiveAvatar", "liveavatar-stock-avatar-poster.jpg", "Sharpest heavyweight identity; restrained expression."),
    ("SoulX-FlashTalk", "soulx-flashtalk-stock-avatar-poster.jpg", "Stable face; clear mouth motion; costly cold compile."),
    ("EchoMimicV3-Flash", "echomimic-v3-flash-stock-avatar-poster.jpg", "Best quality-to-infrastructure balance on L40S."),
    ("SoulX-FlashHead Lite", "soulx-flashhead-stock-avatar-poster.jpg", "Lowest memory; tight crop; softer identity."),
  ]
  cards = []
  for name, filename, caption in poster_info:
    img = Image(
      str(ROOT / "assets/posters" / filename),
      width=1.36 * inch,
      height=2.27 * inch,
      kind="proportional",
    )
    card = Table(
      [[img], [p(f"<b>{name}</b><br/>{caption}", "caption")]],
      colWidths=[1.65 * inch],
      hAlign="CENTER",
    )
    card.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 1), ("RIGHTPADDING", (0, 0), (-1, -1), 1), ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    cards.append(card)
  gallery = Table([cards[:3], cards[3:]], colWidths=[1.92 * inch] * 3, hAlign="LEFT")
  gallery.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D5DD")), ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D5DD")), ("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
  story.append(gallery)
  story.append(p("Posters are shown for print comparison. The repository contains the corresponding muted MP4s and SHA-256 checksums.", "small"))

  story.append(PageBreak())
  story.append(p("6. Model observations", "h1"))
  observations = [
    ("Wan2.2-S2V-14B", "Wan preserved identity and background with conservative facial motion. It was the slowest successful run and cost more than four times LiveAvatar for this short output. Dependency fixes were needed before inference; failed pre-denoising attempts are excluded."),
    ("LongCat-Video-Avatar 1.5", "The eight-step INT8 path produced the most visible performance, including a hand gesture absent from the reference image. It ran on one H200 with context parallelism set to one. Peak allocation leaves little 48 GB operational headroom."),
    ("LiveAvatar", "The four-step FP8 path produced the sharpest identity, convincing mouth shapes, blinking, and restrained expression. It was the fastest H200 job but still needed 61,401 MiB and model offloading during final VAE decode."),
    ("SoulX-FlashTalk", "FlashTalk was visually stable and designed for continuous chunks. First-chunk graph compilation took 119.58 seconds; later chunks took about 3.75 seconds. The cache was the largest in the test."),
    ("EchoMimicV3-Flash", "The 1.3B eight-step path produced clean 25 fps motion on L40S. Its observed 33,726 MiB peak is for the official 768-area configuration, not a theoretical minimum-memory mode."),
    ("SoulX-FlashHead Lite", "FlashHead used 5,763 MiB. First compilation took 155.7 seconds, while later chunks took about 0.19 seconds. The close crop and weaker mouth motion reduced visual polish."),
    ("LTX-2.3 DubIt", "The official checkpoint required approval unavailable to the test environment. The job stopped during CPU-only weight preparation. No unofficial mirror was substituted and no GPU was allocated."),
  ]
  for name, text in observations:
    story.append(p(name, "h2"))
    story.append(p(text))

  story.append(PageBreak())
  story.append(p("7. Voice behavior", "h1"))
  story.append(p("The six completed image-and-audio systems animate supplied speech and mux that speech into their output; they do not select or clone a voice. Correlation with the aligned input ranged from 0.999198 to 0.999897. Small differences are consistent with resampling and AAC encoding rather than creation of a new speaker."))
  voice_rows = [["Model", "Input speech", "Voice learned?", "Correlation"]]
  for item in DATA["results"]:
    if item["model"] == "LTX-2.3 DubIt":
      voice_rows.append([item["model"], "Target text", "From reference video", "Not run"])
    else:
      voice_rows.append([item["model"], "Preserved", "No", f'{item["audio_correlation"]:.6f}'])
  story.append(table(voice_rows, [2.05 * inch, 1.0 * inch, 1.35 * inch, 0.85 * inch], 6.9))
  story.append(p("DubIt is a different interface class: it takes target text and a voiced reference video, aiming to retain vocal identity while synthesizing new speech."))

  story.append(p("8. Deployment interpretation", "h1"))
  for text in [
    "LiveAvatar for the sharpest observed heavyweight output when 80 GB-class GPU capacity is available.",
    "EchoMimicV3-Flash for the best observed balance of result quality, cache size, and 48 GB-class deployment.",
    "LongCat Avatar 1.5 when expressive body motion matters more than conservative motion.",
    "SoulX-FlashHead Lite when low memory and warmed streaming speed dominate visual polish.",
  ]:
    story.append(bullet(text))

  story.append(PageBreak())
  story.append(p("9. Limitations", "h1"))
  limitations = [
    "One portrait and one short English sentence; no demographic, language, emotion, occlusion, profile-view, or long-duration coverage.",
    "No preregistered blinded human study; qualitative judgments are author observations.",
    "Official paths differ in resolution, frame rate, steps, quantization, prompts, and hardware.",
    "Cold-job timing disadvantages compilation-heavy streaming systems.",
    "A 0.5-second VRAM poll can miss shorter allocation spikes.",
    "Modal prices and scheduling are point-in-time cloud observations, not universal total cost of ownership.",
    "Normalized summaries are public, but full internal job logs, cached weights, and the source narration waveform are not.",
    "Access, licenses, stars, dependencies, and upstream recommendations can change.",
  ]
  for item in limitations:
    story.append(bullet(item))
  story.append(p("10. Reproducibility package", "h1"))
  story.append(p("The repository contains exact runner code, pinned upstream commits and checkpoint revisions, CSV and JSON results, SHA-256 checksums, the public reference frame, muted outputs, poster frames, and build/validation scripts. The reproduction guide defines every measurement boundary and provides all Modal commands."))
  story.append(p("A stronger follow-up should repeat multiple seeds, use diverse consented portraits and rights-cleared multilingual voices, separate cold-start from warmed throughput, record failure rate, and run a preregistered blinded perceptual study. It should be published as a new dated dataset rather than overwrite this cohort."))

  story.append(p("11. Conclusion", "h1"))
  story.append(p("Recent open-source talking-avatar pipelines can generate convincing short portrait videos from a single image, but deployment requirements differ by an order of magnitude. LiveAvatar led heavyweight visual quality in this test. EchoMimic came close while using a cheaper GPU and less than half the model storage of the large generators. FlashHead showed that a complete generator can fit well below 16 GB, with visible quality tradeoffs. For serving decisions, checkpoint size, memory headroom, warm-up behavior, and access terms matter as much as the strongest sample frame."))

  story.append(p("References", "h1"))
  references = [
    "Wan-Video, <a href='https://github.com/Wan-Video/Wan2.2'>Wan2.2</a>.",
    "Lightricks, <a href='https://github.com/Lightricks/LTX-2'>LTX-2</a> and DubIt guide.",
    "Meituan, <a href='https://github.com/meituan-longcat/LongCat-Video'>LongCat-Video</a>.",
    "Alibaba Quark, <a href='https://github.com/Alibaba-Quark/LiveAvatar'>LiveAvatar</a>.",
    "Soul AI Lab, SoulX-FlashTalk and SoulX-FlashHead.",
    "Ant Group, <a href='https://github.com/antgroup/echomimic_v3'>EchoMimicV3</a>.",
    "Pexels, reference video by AI25.Studio and Pexels License.",
    "Modal, pricing observed August 23, 2026.",
    "Apple, macOS Software License Agreement, System Voices section.",
  ]
  for index, reference in enumerate(references, start=1):
    story.append(p(f"{index}. {reference}", "small"))
  story.append(Spacer(1, 0.12 * inch))
  story.append(p("Repository: <a href='https://github.com/tight-studio/open-source-talking-avatar-benchmark'>github.com/tight-studio/open-source-talking-avatar-benchmark</a>", "callout"))
  return story


def main() -> None:
  OUTPUT.parent.mkdir(parents=True, exist_ok=True)
  document = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=letter,
    rightMargin=0.68 * inch,
    leftMargin=0.68 * inch,
    topMargin=0.62 * inch,
    bottomMargin=0.66 * inch,
    title="A Deployment Benchmark of Popular Open-Source Talking-Avatar Models Released in 2025-2026",
    author="Ethan Jiang and Tight Studio",
    subject="Open-source talking-avatar deployment benchmark",
  )
  document.build(build_story(), onFirstPage=footer, onLaterPages=footer)
  print(OUTPUT)


if __name__ == "__main__":
  main()
