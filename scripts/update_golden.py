#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import check_exact


DEFAULT_MANIFEST = ROOT / "tests" / "golden" / "manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update or check frozen native CPU exactness goldens.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST, help="Golden manifest path to write or check")
    parser.add_argument("--check", action="store_true", help="Fail if the golden manifest is out of date")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = check_exact.golden_manifest()
    path = args.output

    if args.check:
        expected = check_exact._load_manifest(path)
        messages = check_exact._compare_entries(manifest, expected)
        if messages:
            print(f"exactness mismatch against {path}", file=sys.stderr)
            for message in messages[:10]:
                print(message, file=sys.stderr)
            return 1
        print(f"Golden exactness manifest is current: {path}")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote golden exactness manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
