# Contributing

Corrections and reproducibility improvements are welcome. Please keep changes evidence-backed and scoped to the benchmark.

- Do not replace recorded measurements with estimates.
- Add a source and observation date for changing metadata such as stars, prices, or model access.
- Keep upstream commits and checkpoint revisions pinned.
- Do not add portraits, speech, or videos without clear redistribution rights and consent.
- Run `python3 scripts/validate_repository.py` before opening a pull request.

New benchmark runs should include the exact runner revision, inputs, GPU, configuration, raw summary JSON, and output checksum. A new model should not be inserted retroactively into the August 23, 2026 cohort; publish it as a dated extension instead.
