"""Validate Mull's SQLite resource-range audit without accepting empty reports."""

import argparse
import json
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
    1: "killed",
    2: "survived",
    3: "timed_out",
    4: "crashed",
    5: "abnormal_exit",
}
NOT_COVERED = 8


def summarize(counts: Counter) -> dict:
    denominator = counts[1] + counts[2]
    return {
        "total": sum(counts.values()),
        **{label: counts[status] for status, label in EVALUATED_STATUSES.items()},
        "not_covered": counts[NOT_COVERED],
        "mutation_score": {
            "definition": "killed / (killed + survived); excludes timeouts, crashes, abnormal exits and uncovered mutants",
            "denominator": denominator,
            "percent": 100 * counts[1] / denominator if denominator else None,
        },
    }


def validate(report: Path, expected_source: Path, repository_root: Path | None = None) -> dict:
    if not report.is_file():
        raise ValueError(f"Mull report is missing: {report}")
    expected_source = expected_source.resolve(strict=True)
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[1]
    repository_root = repository_root.resolve(strict=True)
    source_key = expected_source.relative_to(repository_root).as_posix()
    counts = Counter()
    files = {}
    mutators = {}
    with closing(sqlite3.connect(report.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        rows = connection.execute("SELECT filename, directory, execution_status, mutator FROM mutant")
        for filename, directory, status, mutator in rows:
            source = Path(filename)
            if not source.is_absolute():
                if not directory or not Path(directory).is_absolute():
                    raise ValueError(f"Cannot resolve mutant source: {filename!r}, {directory!r}")
                source = Path(directory) / source
            if source.resolve() != expected_source:
                raise ValueError(f"Unexpected mutated source: {source}")
            if status not in EVALUATED_STATUSES and status != NOT_COVERED:
                raise ValueError(f"Unsupported or unevaluated Mull status: {status!r}")
            if not isinstance(mutator, str) or not mutator:
                raise ValueError(f"Invalid Mull mutator: {mutator!r}")
            counts[status] += 1
            files.setdefault(source_key, Counter())[status] += 1
            mutators.setdefault(mutator, Counter())[status] += 1
    if not counts[1] + counts[2]:
        raise ValueError("Mull report contains no meaningfully evaluated resource-range mutants (killed or survived)")
    return {
        "schema_version": 1,
        **summarize(counts),
        "per_file": {key: summarize(value) for key, value in sorted(files.items())},
        "per_mutator": {key: summarize(value) for key, value in sorted(mutators.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("expected_source", type=Path)
    parser.add_argument("--json-output", type=Path, help="Write a machine-readable summary")
    parser.add_argument("--repository-root", type=Path, help="Root for repository-relative summary paths (defaults to this checkout)")
    args = parser.parse_args()
    try:
        summary = validate(args.report, args.expected_source, args.repository_root)
        if args.json_output:
            args.json_output.write_text(json.dumps(summary, indent=2) + "\n")
    except (ValueError, OSError, sqlite3.Error, TypeError) as error:
        print(f"Mull report validation failed: {error}", file=sys.stderr)
        return 1
    labels = [*EVALUATED_STATUSES.values(), "not_covered"]
    print("Validated Mull report: " + ", ".join(f"{label}={summary[label]}" for label in labels))
    score = summary["mutation_score"]
    print(f"Mutation score: {score['percent']:.2f}% ({summary['killed']}/{score['denominator']}); {score['definition']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
