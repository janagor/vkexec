#!/usr/bin/env python3
"""Check physical line limits for C++ files in the checkout."""

import argparse
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent.parent
SOURCE_EXTENSIONS = {".cpp", ".cc", ".cxx"}
HEADER_EXTENSIONS = {".hpp", ".h", ".hxx"}
TEST_DIRECTORIES = {"test", "tests", "benchmark", "benchmarks"}
EXCLUDED_DIRECTORIES = {
    ".git", "build", "generated", "node_modules", "out", "target", "third_party", "vendor"
}
SOURCE_LIMIT = 500
HEADER_LIMIT = 600
TEST_LIMIT = 800


def limit_for(path: Path) -> int | None:
    if any(part in EXCLUDED_DIRECTORIES for part in path.parts[:-1]):
        return None
    if path.suffix not in SOURCE_EXTENSIONS | HEADER_EXTENSIONS:
        return None
    if any(part in TEST_DIRECTORIES for part in path.parts[:-1]):
        return TEST_LIMIT
    if path.suffix in HEADER_EXTENSIONS:
        return HEADER_LIMIT
    return SOURCE_LIMIT


def count_lines(path: Path) -> int:
    data = path.read_bytes()
    return data.count(b"\n") + (bool(data) and not data.endswith(b"\n"))


def tracked_paths(root: Path) -> list[Path]:
    output = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    )
    return [Path(os.fsdecode(entry)) for entry in output.split(b"\0") if entry]


def check(root: Path) -> list[tuple[Path, int, int]]:
    rows = []
    for relative in tracked_paths(root):
        limit = limit_for(relative)
        absolute = root / relative
        if limit is None or not absolute.is_file() or absolute.is_symlink():
            continue
        lines = count_lines(absolute)
        rows.append((relative, lines, limit))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="Show all files without failing")
    args = parser.parse_args()

    rows = check(ROOT)
    violations = [row for row in rows if row[1] > row[2]]
    selected = rows if args.report else violations
    for path, lines, limit in sorted(selected, key=lambda row: (-row[1], str(row[0]))):
        print(f"{path}: {lines} lines (limit {limit})")
    print(f"Checked {len(rows)} files; {len(violations)} violations")
    return 0 if args.report or not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
