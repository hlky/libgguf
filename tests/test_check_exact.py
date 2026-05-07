from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bench import check_exact
from libgguf import GGMLQuantizationType


ROOT = Path(__file__).resolve().parents[1]

TINY_Q4_K_RESULTS = [
    {
        "qtype": "Q4_K",
        "shape": "1x256",
        "case": "zeros",
        "encoded_nbytes": 144,
        "encoded_sha256": "81c611f35bff79491538b2f7cf201c7597a661a5c549633541c62bdc8af1613f",
        "decoded_all_finite": True,
    },
    {
        "qtype": "Q4_K",
        "shape": "1x256",
        "case": "normal",
        "encoded_nbytes": 144,
        "encoded_sha256": "d0f12207d70a5305169f470ea3540d2b06d71bc715fdf21649f6f4c57cc713f7",
        "decoded_all_finite": True,
    },
]


def _dummy_golden_manifest() -> dict[str, object]:
    return {
        "version": check_exact.GOLDEN_MANIFEST_VERSION,
        "generator": "scripts/update_golden.py",
        "fixture": check_exact.GOLDEN_FIXTURE,
        "qtypes": list(check_exact.GOLDEN_QTYPES),
        "shapes": [check_exact._shape_id(shape) for shape in check_exact.GOLDEN_SHAPES],
        "patterns": list(check_exact.GOLDEN_PATTERNS),
        "entries": [
            {
                "qtype": qtype,
                "shape": check_exact._shape_id(shape),
                "pattern": pattern,
                "encoded_nbytes": 1,
                "encoded_sha256": "0" * 64,
                "decoded_all_finite": True,
            }
            for qtype in check_exact.GOLDEN_QTYPES
            for shape in check_exact.GOLDEN_SHAPES
            for pattern in check_exact.GOLDEN_PATTERNS
        ],
    }


def _run_check_exact(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    # Keep src/ off PYTHONPATH here so normal wheel installs do not shadow the
    # installed native extension when the CLI is exercised in a subprocess.
    pythonpath = [str(ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return subprocess.run(
        [sys.executable, str(ROOT / "bench" / "check_exact.py"), *args],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_parse_shapes_accepts_comma_separated_positive_rows_by_width() -> None:
    assert check_exact._parse_shapes("1x256, 2X512, 3 x 1024") == [(1, 256), (2, 512), (3, 1024)]


@pytest.mark.parametrize(
    "value",
    [
        "",
        " , ",
        "0x256",
        "1x0",
        "-1x256",
        "1x-256",
        "256",
        "1x",
        "x256",
        "1x2x3",
        "rowsxwidth",
    ],
)
def test_parse_shapes_rejects_non_positive_or_bad_shapes(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        check_exact._parse_shapes(value)


@pytest.mark.parametrize("value", ["", " , "])
def test_parse_qtypes_rejects_empty_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        check_exact._parse_qtypes(value)


def test_run_checks_is_deterministic_for_tiny_q4_k_case() -> None:
    qtypes = [GGMLQuantizationType.Q4_K]
    shapes = [(1, 256)]
    cases = ["zeros", "normal"]

    first = [asdict(result) for result in check_exact._run_checks(qtypes, shapes, cases)]
    second = [asdict(result) for result in check_exact._run_checks(qtypes, shapes, cases)]

    assert first == TINY_Q4_K_RESULTS
    assert second == first


def test_golden_patterns_are_stable_and_distinct() -> None:
    positive = check_exact._case_rows("absmax_tie_positive_first", 1, 8, seed=1)
    negative = check_exact._case_rows("absmax_tie_negative_first", 1, 8, seed=1)
    first_normal = check_exact._case_rows("random_normal_seed0", 1, 8, seed=1)
    second_normal = check_exact._case_rows("random_normal_seed0", 1, 8, seed=999)

    np_positive = positive[0, :4].tolist()
    np_negative = negative[0, :4].tolist()
    assert np_positive == [1.5, -1.5, 0.5, -0.5]
    assert np_negative == [-1.5, 1.5, -0.5, 0.5]
    assert np_positive != np_negative
    assert first_normal.tolist() == second_normal.tolist()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.update(version=2),
        lambda manifest: manifest.update(qtypes=["Q4_0"]),
        lambda manifest: manifest.update(patterns=["zeros"]),
        lambda manifest: manifest["entries"].pop(),
        lambda manifest: manifest["entries"].append(dict(manifest["entries"][0])),
        lambda manifest: manifest["entries"][0].update(qtype="Q4_1"),
    ],
)
def test_golden_manifest_validation_rejects_invalid_manifest(mutate) -> None:
    manifest = _dummy_golden_manifest()
    mutate(manifest)

    with pytest.raises(ValueError):
        check_exact._entries_from_manifest(manifest)


def test_frozen_golden_manifest_matches_native_cpu() -> None:
    manifest = ROOT / "tests" / "golden" / "manifest.json"
    observed = check_exact.golden_manifest()
    expected = check_exact._load_manifest(manifest)

    assert check_exact._compare_entries(observed, expected) == []


def test_json_fixture_round_trip_writes_and_compares_expected_results(tmp_path: Path) -> None:
    fixture = tmp_path / "exactness.json"
    write_result = _run_check_exact(
        "--qtypes",
        "Q4_K",
        "--shapes",
        "1x256",
        "--cases",
        "zeros,normal",
        "--write-json",
        str(fixture),
    )

    assert write_result.returncode == 0, write_result.stderr
    assert json.loads(fixture.read_text(encoding="utf-8")) == TINY_Q4_K_RESULTS
    assert "Q4_K 1x256 zeros" in write_result.stdout

    compare_result = _run_check_exact(
        "--qtypes",
        "Q4_K",
        "--shapes",
        "1x256",
        "--cases",
        "zeros,normal",
        "--expect-json",
        str(fixture),
    )

    assert compare_result.returncode == 0, compare_result.stderr
    assert "Q4_K 1x256 normal" in compare_result.stdout


def test_json_fixture_mismatch_fails_clearly(tmp_path: Path) -> None:
    fixture = tmp_path / "exactness.json"
    altered = [dict(result) for result in TINY_Q4_K_RESULTS]
    altered[0]["encoded_sha256"] = "0" * 64
    fixture.write_text(json.dumps(altered), encoding="utf-8")

    result = _run_check_exact(
        "--qtypes",
        "Q4_K",
        "--shapes",
        "1x256",
        "--cases",
        "zeros,normal",
        "--expect-json",
        str(fixture),
    )

    assert result.returncode != 0
    assert f"exactness mismatch against {fixture}" in result.stderr


def test_empty_cases_fail_instead_of_noop() -> None:
    result = _run_check_exact("--qtypes", "Q4_K", "--shapes", "1x256", "--cases", " , ")

    assert result.returncode != 0
    assert "at least one case is required" in result.stderr
