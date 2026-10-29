"""Validate Mull's SQLite resource-range audit without accepting empty reports."""

import argparse
from collections import Counter
from contextlib import closing
from pathlib import Path
import sqlite3
import sys


# Schema and enum inspected at mull-project/mull revision
# dab0b590af1ae1ec34609d993668a9a12a703c30:
# rust/mull-reporters/src/sqlite_reporter.rs and
# rust/mull-state/src/execution_status.rs.
EVALUATED_STATUSES = {
    1: "failed",
    2: "passed",
    3: "timedout",
    4: "crashed",
    5: "abnormal_exit",
}
NOT_COVERED = 8


def validate(report: Path, expected_source: Path) -> Counter:
    if not report.is_file():
        raise ValueError(f"Mull report is missing: {report}")
    expected_source = expected_source.resolve(strict=True)
    counts = Counter()
    with closing(sqlite3.connect(report.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        rows = connection.execute("SELECT filename, directory, execution_status FROM mutant")
        for filename, directory, status in rows:
            source = Path(filename)
            if not source.is_absolute():
                if not directory or not Path(directory).is_absolute():
                    raise ValueError(f"Cannot resolve mutant source: {filename!r}, {directory!r}")
                source = Path(directory) / source
            if source.resolve() != expected_source:
                raise ValueError(f"Unexpected mutated source: {source}")
            if status not in EVALUATED_STATUSES and status != NOT_COVERED:
                raise ValueError(f"Unsupported or unevaluated Mull status: {status!r}")
            counts[status] += 1
    if not sum(counts[status] for status in EVALUATED_STATUSES):
        raise ValueError("Mull report contains no evaluated resource-range mutants")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("expected_source", type=Path)
    args = parser.parse_args()
    try:
        counts = validate(args.report, args.expected_source)
    except (ValueError, OSError, sqlite3.Error, TypeError) as error:
        print(f"Mull report validation failed: {error}", file=sys.stderr)
        return 1
    summary = ", ".join(f"{label}={counts[status]}" for status, label in EVALUATED_STATUSES.items())
    print(f"Validated Mull report: {summary}, not_covered={counts[NOT_COVERED]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
