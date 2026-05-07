# Benchmark Results

`bench/results/` stores curated benchmark and exactness snapshots from benchmark
drivers under `bench/`. These files should be low-level run outputs: CSV/JSON
measurements, short per-run summaries, and focused notes that make a benchmark
result reproducible or comparable.

Use `reports/` for higher-level analysis, policy writeups, GGUF comparisons, or
cross-run conclusions built from these measurements.

## Naming

- Use descriptive, stable names for promoted baselines, such as
  `q4_k_lane_subgroups_production.csv`.
- For one captured run with several files, prefer a directory named
  `<subject>_<YYYYMMDDTHHMMSSZ>/` with `results.csv`, `results.json`, and an
  optional `summary.md`.
- Keep paired CSV/JSON files under the same stem when they describe the same
  measurement.

## Promotion

Commit benchmark artifacts only when they support a code change, release
decision, policy decision, or documented investigation. Promoted artifacts should
have enough context in their name or summary to identify the benchmark subject,
backend, qtypes, and comparison target.

Ad hoc local outputs should stay under ignored names such as
`bench/results/local_*` or `.cache/`. Do not commit machine-specific scratch runs
unless they have been intentionally promoted to a curated baseline.
