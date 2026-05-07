from __future__ import annotations

import json
from pathlib import Path

from bench import conversion_bench


def test_parse_timings_accepts_native_and_variant_units() -> None:
    stderr = (
        "Timings: total=1.250s metadata=10ms read: 250us "
        "cpu_convert=20ms h2d=30us cuda_quant=0.5 seconds d2h=40us "
        "quant=0.75s write=0.125sec tensors=2\n"
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
                "print('Timings: total=1.234s metadata=0.010s read=0.020s cpu_convert=0.030s h2d=0.040s cuda_quant=0.900s d2h=0.050s write=0.200s tensors=3 threads=2 scratch=4096', file=sys.stderr)",
            ]
        ),
        encoding="utf-8",
    )
    fake_converter.chmod(fake_converter.stat().st_mode | 0o111)
    results_root = tmp_path / "results"

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
    assert payload["summary"]["runs"] == 2
    assert payload["summary"]["successful_runs"] == 2
    assert payload["summary"]["total_s_mean"] == 1.234
    assert payload["results"][0]["cpu_convert_s"] == 0.03
    assert payload["results"][0]["h2d_s"] == 0.04
    assert payload["results"][0]["cuda_quant_s"] == 0.9
    assert payload["results"][0]["d2h_s"] == 0.05
    assert payload["results"][0]["qtype_counts"] == {"Q4_K": 2, "Q8_0": 1}
    assert payload["results"][0]["output_size_bytes"] == len(b"GGUF fake output")
    assert "--timings" in payload["results"][0]["command"]
    assert "--threads" in payload["results"][0]["command"]
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "Q4_K" in csv_text
    assert "cpu_convert_s" in csv_text
    assert "cuda_quant_s" in csv_text


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
