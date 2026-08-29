#!/usr/bin/env python3
"""Validate benchmark records, checksums, runners, and public media tracks."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = {
  "Wan2.2-S2V-14B",
  "LTX-2.3 DubIt",
  "LongCat-Video-Avatar 1.5",
  "LiveAvatar",
  "SoulX-FlashTalk",
  "EchoMimicV3-Flash",
  "SoulX-FlashHead Lite",
}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def load_checksums() -> dict[str, str]:
  checksums: dict[str, str] = {}
  for line in (ROOT / "CHECKSUMS.sha256").read_text().splitlines():
    if not line.strip():
      continue
    expected, relative = line.split(maxsplit=1)
    checksums[relative.strip()] = expected
  return checksums


def validate_data() -> list[dict[str, str]]:
  with (ROOT / "data/results.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
  assert {row["model"] for row in rows} == EXPECTED_MODELS
  assert len(rows) == 7

  payload = json.loads((ROOT / "data/results.json").read_text())
  assert payload["schema_version"] == "1.0.0"
  assert {item["model"] for item in payload["results"]} == EXPECTED_MODELS
  assert sum(item["status"] == "completed" for item in payload["results"]) == 6
  assert sum(item["status"] == "blocked" for item in payload["results"]) == 1
  return rows


def validate_checksums() -> None:
  for relative, expected in load_checksums().items():
    path = ROOT / relative
    assert path.is_file(), f"missing artifact: {relative}"
    actual = sha256(path)
    assert actual == expected, f"checksum mismatch: {relative}"


def validate_paths(rows: list[dict[str, str]]) -> None:
  videos = []
  for row in rows:
    runner = ROOT / row["runner"]
    assert runner.is_file(), f"missing runner: {runner}"
    if row["status"] == "completed":
      video = ROOT / row["video"]
      assert video.is_file() and video.stat().st_size > 0
      videos.append(video)
  assert len(videos) == 6


def validate_python() -> None:
  runners = sorted((ROOT / "runners/modal").glob("*.py"))
  result = subprocess.run(
    [sys.executable, "-m", "py_compile", *map(str, runners)],
    check=False,
    capture_output=True,
    text=True,
  )
  assert result.returncode == 0, result.stderr


def main() -> None:
  rows = validate_data()
  validate_paths(rows)
  validate_checksums()
  validate_python()
  print("Validated 7 model records, 6 public videos, runners, and checksums.")


if __name__ == "__main__":
  main()
