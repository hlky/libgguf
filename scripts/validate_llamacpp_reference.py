"""Validate vendored GGML quant/dequant reference code against llama.cpp.

The local C++ sources split selected scalar reference functions from upstream
``ggml/src/ggml-quants.c`` into per-qtype files and add libgguf dispatch glue.
This checker compares the copied algorithm bodies, while normalizing the small
C-to-C++ adapter differences that are intentionally local to libgguf:

* ``quantize_row_*_ref`` upstream names become local ``quantize_row_*`` names;
* ``GGML_ASSERT``/diagnostic abort calls are ignored for algorithm matching;
* ``NULL``/``nullptr`` and required C++ casts from ``void *`` are treated alike.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


EXPECTED_LLAMA_CPP_COMMIT = "dbe9c0c8ce65354c372f5d4ab507e5424a755e9f"
LLAMA_CPP_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"
UPSTREAM_QUANTS_PATH = Path("ggml/src/ggml-quants.c")


@dataclass(frozen=True)
class FunctionCheck:
    local_file: Path
    local_name: str
    upstream_name: str


def _run_git(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def _extract_function(text: str, name: str) -> str | None:
    text = _strip_c_comments(text)
    match = re.search(
        r"(^|\n)([^;{}#\n]*\b" + re.escape(name) + r"\s*\([^;{}]*\)\s*)\{",
        text,
    )
    if not match:
        return None

    start = match.end() - 1
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[match.start(2) : index + 1]

    raise ValueError(f"unterminated function body for {name}")


def _remove_call_statements(text: str, names: tuple[str, ...]) -> str:
    for name in names:
        output: list[str] = []
        index = 0
        pattern = f"{name}("
        while index < len(text):
            start = text.find(pattern, index)
            if start < 0:
                output.append(text[index:])
                break
            if start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
                output.append(text[index : start + len(pattern)])
                index = start + len(pattern)
                continue

            output.append(text[index:start])
            cursor = start + len(pattern)
            depth = 1
            in_string = False
            escaped = False
            while cursor < len(text) and depth:
                char = text[cursor]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                elif char == '"':
                    in_string = True
                elif char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                cursor += 1

            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor < len(text) and text[cursor] == ";":
                cursor += 1
            index = cursor
        text = "".join(output)
    return text


def _canonical_function(text: str) -> str:
    text = _strip_c_comments(text)
    text = text.replace("GGML_RESTRICT", "RESTRICT")
    text = text.replace("GGML_ASSERT", "assert")
    text = text.replace("NULL", "nullptr")
    text = re.sub(r"quantize_row_([A-Za-z0-9_]+)_ref", r"quantize_row_\1", text)
    text = text.replace("ggml_row_size", "libgguf_row_size")
    text = _remove_call_statements(text, ("assert", "GGML_ABORT", "abort", "printf"))
    text = re.sub(r"=\s*\(block_[A-Za-z0-9_]+\s*\*\)\s*vy", "=vy", text)
    text = re.sub(r"=\s*\(const\s+block_[A-Za-z0-9_]+\s*\*\)\s*vx", "=vx", text)
    return re.sub(r"\s+", "", text)


def _collect_checks(repo_root: Path) -> list[FunctionCheck]:
    checks: list[FunctionCheck] = []

    for path in sorted((repo_root / "csrc/quant").glob("*.cpp")):
        text = _strip_c_comments(path.read_text(encoding="utf-8"))
        for match in re.finditer(r"\b(?:void|static\s+void)\s+(quantize_row_[A-Za-z0-9_]+)\s*\(", text):
            local_name = match.group(1)
            upstream_name = local_name if local_name.endswith("_impl") else f"{local_name}_ref"
            checks.append(FunctionCheck(path.relative_to(repo_root), local_name, upstream_name))

    dequant_path = repo_root / "csrc/dequant/dequant.cpp"
    dequant_text = _strip_c_comments(dequant_path.read_text(encoding="utf-8"))
    for match in re.finditer(r"\bvoid\s+(dequantize_row_[A-Za-z0-9_]+)\s*\(", dequant_text):
        name = match.group(1)
        checks.append(FunctionCheck(dequant_path.relative_to(repo_root), name, name))

    return checks


def _checkout_upstream(args: argparse.Namespace) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if args.llama_cpp:
        return Path(args.llama_cpp).resolve(), None

    temp_dir = tempfile.TemporaryDirectory(prefix="libgguf-llama-cpp-")
    checkout = Path(temp_dir.name) / "llama.cpp"
    ref = args.ref or args.expected_commit
    _run_git(["clone", "--quiet", "--filter=blob:none", "--no-checkout", args.repo_url, str(checkout)])
    _run_git(["checkout", "--quiet", ref], cwd=checkout)
    return checkout, temp_dir


def _commit_metadata(checkout: Path) -> tuple[str | None, str | None, str | None]:
    try:
        commit = _run_git(["rev-parse", "HEAD"], cwd=checkout)
        date = _run_git(["show", "-s", "--format=%cI", "HEAD"], cwd=checkout)
        subject = _run_git(["show", "-s", "--format=%s", "HEAD"], cwd=checkout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None, None
    return commit, date, subject


def _validate_commit(args: argparse.Namespace, checkout: Path) -> tuple[str | None, str | None, str | None]:
    commit, date, subject = _commit_metadata(checkout)
    if not commit:
        if args.allow_different_commit:
            return commit, date, subject
        raise SystemExit("llama.cpp path is not a git checkout; pass --allow-different-commit to skip the pin check")

    if args.expected_commit and commit != args.expected_commit and not args.allow_different_commit:
        raise SystemExit(
            "llama.cpp checkout is at "
            f"{commit}, expected {args.expected_commit}. "
            "Pass --allow-different-commit when intentionally checking current upstream."
        )
    return commit, date, subject


def validate(repo_root: Path, upstream_root: Path, verbose: bool = False) -> int:
    upstream_file = upstream_root / UPSTREAM_QUANTS_PATH
    if not upstream_file.exists():
        raise SystemExit(f"missing upstream source: {upstream_file}")

    upstream_text = upstream_file.read_text(encoding="utf-8")
    checks = _collect_checks(repo_root)
    failures: list[str] = []

    for check in checks:
        local_text = (repo_root / check.local_file).read_text(encoding="utf-8")
        local_func = _extract_function(local_text, check.local_name)
        upstream_func = _extract_function(upstream_text, check.upstream_name)

        label = f"{check.local_file.as_posix()}::{check.local_name} <= {check.upstream_name}"
        if local_func is None:
            failures.append(f"missing local function: {label}")
            continue
        if upstream_func is None:
            failures.append(f"missing upstream function: {label}")
            continue
        if _canonical_function(local_func) != _canonical_function(upstream_func):
            failures.append(f"mismatch: {label}")
        elif verbose:
            print(f"match: {label}")

    if failures:
        print("llama.cpp reference validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"Validated {len(checks)} vendored reference functions against {UPSTREAM_QUANTS_PATH.as_posix()}.")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp", type=Path, help="Path to an existing ggml-org/llama.cpp checkout.")
    parser.add_argument(
        "--expected-commit",
        default=EXPECTED_LLAMA_CPP_COMMIT,
        help="Pinned llama.cpp commit to require unless --allow-different-commit is passed.",
    )
    parser.add_argument("--ref", help="Git ref to checkout when --llama-cpp is not supplied.")
    parser.add_argument("--repo-url", default=LLAMA_CPP_REPO_URL, help="llama.cpp Git URL used for temporary checkout.")
    parser.add_argument(
        "--allow-different-commit",
        action="store_true",
        help="Compare against the supplied checkout/ref even if it does not match --expected-commit.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each matched function.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    checkout, temp_dir = _checkout_upstream(args)
    try:
        commit, date, subject = _validate_commit(args, checkout)
        result = validate(repo_root, checkout, args.verbose)
        if result == 0 and commit:
            print(f"llama.cpp commit: {commit}")
            if date:
                print(f"commit date: {date}")
            if subject:
                print(f"commit subject: {subject}")
        return result
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
