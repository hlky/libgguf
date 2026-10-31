"""Compile production HIP quantizer objects without loading or executing them."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--architectures', nargs='+', choices=('gfx942', 'gfx1201', 'gfx1151'),
                        default=['gfx942', 'gfx1201'])
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    sdk, output = args.sdk.resolve(), args.output.resolve()
    clang = sdk / 'lib/llvm/bin/clang++'
    if clang.with_suffix('.exe').is_file():
        clang = clang.with_suffix('.exe')
    output.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output([str(clang), '--version'], text=True)
    headers = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted((ROOT / 'include').rglob('*')) if path.is_file()}

    def compile_one(architecture, source):
        directory = output / architecture
        directory.mkdir(parents=True, exist_ok=True)
        obj = directory / (source.stem + '.obj')
        command = [str(clang), '-x', 'hip', '--offload-arch=' + architecture,
                   '--rocm-path=' + str(sdk),
                   '--rocm-device-lib-path=' + str(sdk / 'lib/llvm/amdgcn/bitcode'),
                   '-D__HIP_PLATFORM_AMD__=1', '-std=c++17', '-O2', '-ffp-contract=off',
                   '-I' + str(ROOT / 'include'), '-c', str(source), '-o', str(obj)]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        (directory / (source.stem + '.log')).write_text(result.stdout + result.stderr, encoding='utf-8')
        return dict(architecture=architecture, source=str(source.relative_to(ROOT)),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    command=command, exit_code=result.returncode,
                    object_sha256=hashlib.sha256(obj.read_bytes()).hexdigest() if result.returncode == 0 else None)

    sources = sorted((ROOT / 'csrc/kernels/gpu/quantize').glob('*.cu'))
    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(compile_one, arch, source)
                   for arch in dict.fromkeys(args.architectures) for source in sources]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(record['architecture'], record['source'], record['exit_code'], flush=True)
    report = dict(compiler=version, sdk=str(sdk), qualification_status='compile-only',
                  gpu_execution=False, header_sha256=headers,
                  objects=sorted(records, key=lambda record: (record['architecture'], record['source'])))
    (output / 'compile.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return int(any(record['exit_code'] for record in records))


if __name__ == '__main__':
    raise SystemExit(main())
