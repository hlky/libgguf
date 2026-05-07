# Reports

`reports/` stores curated research and analysis artifacts: GGUF comparison
summaries, qtype inventories, policy notes, and conclusions that combine or
interpret lower-level measurements.

Use `bench/results/` for raw benchmark and exactness run outputs from benchmark
drivers. Reports may reference those results, but should not duplicate large raw
tables unless the table is part of the reviewed analysis.

## Naming

- Use descriptive filenames or subdirectories named after the investigation, for
  example `gguf_comparison/` or `gguf_qtype_inventory.md`.
- Keep Markdown summaries and machine-readable JSON sidecars under matching
  stems when they describe the same finding.
- Prefer stable names for living baselines and dated names only when preserving a
  specific point-in-time comparison matters.

## Promotion

Commit reports only when they support a code change, release decision, policy
decision, or documented investigation. A promoted report should state the
question it answers, the source data or comparison target, and the conclusion or
next action.

Ad hoc local analysis should stay in ignored space such as `reports/exactness/`,
`bench/results/local_*`, or `.cache/` until it is intentionally promoted.
