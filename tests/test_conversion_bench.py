from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench import conversion_bench


def test_parse_timings_accepts_native_and_variant_units() -> None:
    stderr = (
        "Timings: total=1.250s metadata=10ms read: 250us "
        "cpu_convert=20ms h2d=30us cuda_quant=0.5 seconds d2h=40us "
        "quant=0.75s write=0.125sec tensors=2 "
        "cuda_chunks=3 cuda_vram=1073741824 cuda_max_input=264241152 cuda_max_output=37158912\n"
    )

    assert conversion_bench.parse_timings(stderr) == {
        "total_s": 1.25,
        "metadata_s": 0.01,
        "read_s": 0.00025,
        "cpu_convert_s": 0.02,
        "h2d_s": 0.00003,
        "cuda_quant_s": 0.5,
        "d2h_s": 0.00004,
        "quant_s": 0.75,
        "write_s": 0.125,
        "cuda_chunks": 3,
        "cuda_vram_bytes": 1073741824,
        "cuda_max_input_bytes": 264241152,
        "cuda_max_output_bytes": 37158912,
    }


def test_parse_qtype_and_fallback_counts() -> None:
    stdout = "Wrote model.gguf\nTensor types: Q4_K=2, Q8_0=1\nFallbacks: F16=3\n"

    assert conversion_bench.parse_key_value_counts(stdout, "Tensor types:") == {"Q4_K": 2, "Q8_0": 1}
    assert conversion_bench.parse_key_value_counts(stdout, "Fallbacks:") == {"F16": 3}


def test_main_runs_fake_converter_and_writes_json_csv(tmp_path: Path) -> None:
    src = tmp_path / "model.safetensors"
    src.write_bytes(b"fake")
    fake_converter = tmp_path / "fake_converter.py"
    fake_converter.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from pathlib import Path",
                "import sys",
                "args = sys.argv[1:]",
                "dst = Path(args[args.index('--dst') + 1])",
                "dst.write_bytes(b'GGUF fake output')",
                "print(f'Wrote {dst}')",
                "print('Architecture: flux')",
                "print('File type: Q4_K_M')",
                "print('Tensor types: Q4_K=2, Q8_0=1')",
                "print('Timings: total=1.234s metadata=0.010s read=0.020s cpu_convert=0.030s h2d=0.040s cuda_quant=0.900s d2h=0.050s write=0.200s tensors=3 cuda_chunks=5 threads=2 scratch=4096 cuda_vram=8192 cuda_max_input=4096 cuda_max_output=1024', file=sys.stderr)",
            ]
        ),
        encoding="utf-8",
    )
    fake_converter.chmod(fake_converter.stat().st_mode | 0o111)
    results_root = tmp_path / "results"
    output_root = tmp_path / "outputs"

    code = conversion_bench.main(
        [
            "--src",
            str(src),
            "--qtype",
            "Q4_K_M",
            "--converter",
            str(fake_converter),
            "--policy",
            "dynamic",
            "--runs",
            "2",
            "--threads",
            "2",
            "--scratch-bytes",
            "4096",
            "--results-root",
            str(results_root),
            "--output-root",
            str(output_root),
            "--delete-outputs",
            "--run-name",
            "local_fake",
        ]
    )

    assert code == 0
    summary_path = results_root / "local_fake" / "summary.json"
    csv_path = results_root / "local_fake" / "summary.csv"
    assert summary_path.exists()
    assert csv_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["config"]["qtype"] == "Q4_K_M"
    assert payload["config"]["output_dir"] == str(output_root / "local_fake")
    assert payload["config"]["delete_outputs"] is True
    assert payload["summary"]["runs"] == 2
    assert payload["summary"]["successful_runs"] == 2
    assert payload["summary"]["total_s_mean"] == 1.234
    assert payload["results"][0]["cpu_convert_s"] == 0.03
    assert payload["results"][0]["h2d_s"] == 0.04
    assert payload["results"][0]["cuda_quant_s"] == 0.9
    assert payload["results"][0]["d2h_s"] == 0.05
    assert payload["results"][0]["cuda_vram_bytes"] == 8192
    assert payload["results"][0]["cuda_max_input_bytes"] == 4096
    assert payload["results"][0]["cuda_max_output_bytes"] == 1024
    assert payload["results"][0]["cuda_chunks"] == 5
    assert payload["results"][0]["qtype_counts"] == {"Q4_K": 2, "Q8_0": 1}
    assert payload["results"][0]["output_size_bytes"] == len(b"GGUF fake output")
    assert payload["results"][0]["output_deleted"] is True
    assert not Path(payload["results"][0]["dst"]).exists()
    assert Path(payload["results"][0]["dst"]).parent == output_root / "local_fake"
    assert "--timings" in payload["results"][0]["command"]
    assert "--threads" in payload["results"][0]["command"]
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "Q4_K" in csv_text
    assert "cpu_convert_s" in csv_text
    assert "cuda_quant_s" in csv_text
    assert "cuda_vram_bytes" in csv_text
    assert "cuda_chunks" in csv_text
    assert "output_deleted" in csv_text


def test_main_appends_converter_args_after_separator(tmp_path: Path) -> None:
    src = tmp_path / "model.safetensors"
    src.write_bytes(b"fake")
    argv_path = tmp_path / "converter_argv.json"
    fake_converter = tmp_path / "fake_converter.py"
    fake_converter.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                "from pathlib import Path",
                f"argv_path = Path({str(argv_path)!r})",
                "args = sys.argv[1:]",
                "argv_path.write_text(json.dumps(args), encoding='utf-8')",
                "dst = Path(args[args.index('--dst') + 1])",
                "dst.write_bytes(b'GGUF fake output')",
                "print('Timings: total=1s', file=sys.stderr)",
            ]
        ),
        encoding="utf-8",
    )
    fake_converter.chmod(fake_converter.stat().st_mode | 0o111)

    code = conversion_bench.main(
        [
            "--src",
            str(src),
            "--qtype",
            "Q4_0",
            "--converter",
            str(fake_converter),
            "--backend",
            "cuda",
            "--converter-arg=--legacy-flag",
            "--converter-arg=legacy-value",
            "--results-root",
            str(tmp_path / "results"),
            "--run-name",
            "passthrough",
            "--",
            "--backend",
            "cuda",
            "--verify-cuda-tensors",
            "1",
        ]
    )

    assert code == 0
    converter_argv = json.loads(argv_path.read_text(encoding="utf-8"))
    assert converter_argv[-6:] == [
        "--legacy-flag",
        "legacy-value",
        "--backend",
        "cuda",
        "--verify-cuda-tensors",
        "1",
    ]
    payload = json.loads((tmp_path / "results" / "passthrough" / "summary.json").read_text(encoding="utf-8"))
    assert payload["config"]["backend"] == "cuda"
    assert payload["config"]["extra_args"] == converter_argv[-6:]
    assert payload["results"][0]["command"][-6:] == converter_argv[-6:]


def test_main_returns_failure_for_failed_converter(tmp_path: Path) -> None:
    src = tmp_path / "model.safetensors"
    src.write_bytes(b"fake")
    fake_converter = tmp_path / "failed_converter.py"
    fake_converter.write_text(
        "#!/usr/bin/env python3\nimport sys\nprint('nope', file=sys.stderr)\nsys.exit(7)\n",
        encoding="utf-8",
    )
    fake_converter.chmod(fake_converter.stat().st_mode | 0o111)

    code = conversion_bench.main(
        [
            "--src",
            str(src),
            "--qtype",
            "Q8_0",
            "--converter",
            str(fake_converter),
            "--results-root",
            str(tmp_path / "results"),
            "--run-name",
            "failed",
        ]
    )

    assert code == 1
    payload = json.loads((tmp_path / "results" / "failed" / "summary.json").read_text(encoding="utf-8"))
    assert payload["summary"]["failed_runs"] == 1
    assert payload["results"][0]["returncode"] == 7
    assert payload["results"][0]["error"] == "nope"


def write_aggregate(path: Path, *, config: dict[str, object], results: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"config": config, "results": results}), encoding="utf-8")
    return path


def test_compare_writes_markdown_and_joined_aggregate_with_speedups(tmp_path: Path) -> None:
    cpu_path = write_aggregate(
        tmp_path / "cpu.json",
        config={"src": "/models/flux.safetensors", "policy": "dynamic", "runs_per_qtype": 1},
        results=[
            {
                "qtype": "Q4_K_M",
                "total_s": 40.0,
                "encode_s": 30.0,
                "read_s": 5.0,
                "write_s": 2.0,
                "output_size_bytes": 4_000_000_000,
                "output_deleted": True,
            },
            {
                "qtype": "Q2_K",
                "total_s": 10.0,
                "encode_s": 6.0,
                "read_s": 1.0,
                "write_s": 1.0,
                "output_size_bytes": 2_000_000_000,
                "output_deleted": False,
            },
        ],
    )
    cuda_path = write_aggregate(
        tmp_path / "cuda.json",
        config={"policy": "dynamic", "runs_per_qtype": 1},
        results=[
            {
                "qtype": "Q8_0",
                "total_s": 3.0,
                "encode_s": 1.5,
                "read_s": 0.5,
                "write_s": 0.25,
                "output_size_bytes": 8_000_000_000,
                "output_deleted": True,
            },
            {
                "qtype": "Q4_K_M",
                "total_s": 8.0,
                "encode_s": 3.0,
                "read_s": 1.25,
                "write_s": 0.75,
                "output_size_bytes": 4_000_000_000,
                "output_deleted": True,
            },
            {
                "qtype": "Q3_K_M",
                "total_s": 4.0,
                "encode_s": 2.0,
                "read_s": 0.75,
                "write_s": 0.5,
                "output_size_bytes": 3_000_000_000,
                "output_deleted": True,
            },
            {
                "qtype": "Q2_K",
                "total_s": 0.0,
                "encode_s": 0.0,
                "read_s": 0.25,
                "write_s": 0.25,
                "output_size_bytes": 2_000_000_000,
                "output_deleted": True,
            },
        ],
    )
    out_path = tmp_path / "comparison" / "summary.md"

    code = conversion_bench.main(["compare", "--cpu", str(cpu_path), "--cuda", str(cuda_path), "--out", str(out_path)])

    assert code == 0
    joined = json.loads((out_path.parent / "aggregate.json").read_text(encoding="utf-8"))
    assert [row["qtype"] for row in joined["results"]] == ["Q4_K_M", "Q2_K", "Q3_K_M", "Q8_0"]
    assert joined["config"]["cpu_results"] == str(cpu_path)
    assert joined["config"]["cuda_results"] == str(cuda_path)
    assert joined["config"]["src"] == "/models/flux.safetensors"
    assert joined["results"][0]["total_speedup_cuda_vs_cpu"] == 5.0
    assert joined["results"][0]["encode_speedup_cuda_vs_cpu"] == 10.0
    assert joined["results"][0]["output_size_gb"] == 4.0
    assert joined["results"][1]["total_speedup_cuda_vs_cpu"] is None
    assert joined["results"][2]["cpu_total_s"] is None
    assert joined["results"][2]["cuda_total_s"] == 4.0

    markdown = out_path.read_text(encoding="utf-8")
    assert markdown.startswith("# CPU vs CUDA Conversion Qtype Comparison\n")
    assert "| `Q4_K_M` | 40 | 8 | 5x | 30 | 3 | 10x | 4 |" in markdown
    assert "| `Q2_K` | 10 | 0 |  | 6 | 0 |  | 2 |" in markdown
    assert "| `Q3_K_M` |  | 4 |  |  | 2 |  | 3 |" in markdown
    assert "- CPU aggregate: `../cpu.json`" in markdown
    assert "- CUDA aggregate: `../cuda.json`" in markdown


def test_compare_rejects_invalid_aggregate_missing_qtype(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cpu_path = write_aggregate(tmp_path / "cpu.json", config={}, results=[{"total_s": 1.0}])
    cuda_path = write_aggregate(tmp_path / "cuda.json", config={}, results=[])

    with pytest.raises(SystemExit) as exc_info:
        conversion_bench.main(["compare", "--cpu", str(cpu_path), "--cuda", str(cuda_path), "--out", str(tmp_path / "summary.md")])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "missing qtype" in captured.err
