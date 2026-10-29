"""Validate Mull's SQLite mutation scope audit without accepting empty reports."""

import argparse
import json
from collections import Counter
from collections.abc import Sequence
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
    total = sum(counts.values())
    evaluated = sum(counts[status] for status in EVALUATED_STATUSES)
    return {
        "total": total,
        "evaluation_coverage": {
            "definition": "evaluated / discovered; diagnostic only, not a mutation score",
            "evaluated": evaluated,
            "discovered": total,
            "percent": 100 * evaluated / total if total else None,
        },
        **{label: counts[status] for status, label in EVALUATED_STATUSES.items()},
        "not_covered": counts[NOT_COVERED],
        "mutation_score": {
            "definition": "killed / (killed + survived); excludes timeouts, crashes, abnormal exits and uncovered mutants",
            "denominator": denominator,
            "percent": 100 * counts[1] / denominator if denominator else None,
        },
    }


def parse_report(report: Path, expected_sources: Path | Sequence[Path], repository_root: Path | None = None) -> dict:
    """Read and check report structure without requiring evaluated mutants."""
    if not report.is_file():
        raise ValueError(f"Mull report is missing: {report}")
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[1]
    repository_root = repository_root.resolve(strict=True)
    if isinstance(expected_sources, Path):
        expected_sources = [expected_sources]
    sources = {}
    for source in expected_sources:
        resolved = source.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"Expected source is not a file: {source}")
        key = resolved.relative_to(repository_root).as_posix()
        if resolved in sources:
            raise ValueError(f"Duplicate expected source: {source}")
        sources[resolved] = key
    if not sources:
        raise ValueError("Expected source collection must not be empty")
    counts = Counter()
    files = {}
    mutators = {}
    locations = []
    with closing(sqlite3.connect(report.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        rows = connection.execute("SELECT filename, directory, execution_status, mutator, line_number, column_number, mutant_id FROM mutant")
        for filename, directory, status, mutator, line, column, mutant_id in rows:
            source = Path(filename)
            if not source.is_absolute():
                if not directory or not Path(directory).is_absolute():
                    raise ValueError(f"Cannot resolve mutant source: {filename!r}, {directory!r}")
                source = Path(directory) / source
            if source.resolve() not in sources:
                raise ValueError(f"Unexpected mutated source: {source}")
            source_key = sources[source.resolve()]
            if status not in EVALUATED_STATUSES and status != NOT_COVERED:
                raise ValueError(f"Unsupported or unevaluated Mull status: {status!r}")
            if not isinstance(mutator, str) or not mutator:
                raise ValueError(f"Invalid Mull mutator: {mutator!r}")
            if type(line) is not int or type(column) is not int or line < 1 or column < 1:
                raise ValueError(f"Invalid mutant location: {line!r}:{column!r}")
            if not isinstance(mutant_id, str) or not mutant_id.strip():
                raise ValueError(f"Invalid Mull mutant identifier: {mutant_id!r}")
            locations.append({"mutant_id": mutant_id, "file": source_key, "line": line, "column": column,
                              "mutator": mutator,
                              "status": EVALUATED_STATUSES.get(status, "not_covered")})
            counts[status] += 1
            files.setdefault(source_key, Counter())[status] += 1
            mutators.setdefault(mutator, Counter())[status] += 1
    if not counts:
        raise ValueError("Mull report contains no mutants")
    missing = set(sources.values()) - files.keys()
    if missing:
        raise ValueError(f"Expected sources have no mutants: {sorted(missing)}")
    return {
        "schema_version": 2,
        **summarize(counts),
        "locations": sorted(locations, key=lambda row: (row["file"], row["line"], row["column"], row["mutator"], row["status"], row["mutant_id"])),
        "per_file": {key: summarize(value) for key, value in sorted(files.items())},
        "per_mutator": {key: summarize(value) for key, value in sorted(mutators.items())},
    }


def validate(report: Path, expected_sources: Path | Sequence[Path], repository_root: Path | None = None) -> dict:
    """Require a meaningful audit in addition to a structurally valid report."""
    summary = parse_report(report, expected_sources, repository_root)
    if not summary["killed"] + summary["survived"]:
        raise ValueError("Mull report contains no meaningfully evaluated mutants (killed or survived)")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("expected_sources", type=Path, nargs="+")
    parser.add_argument("--json-output", type=Path, help="Write a machine-readable summary")
    parser.add_argument("--repository-root", type=Path, help="Root for repository-relative summary paths (defaults to this checkout)")
    args = parser.parse_args()
    try:
        summary = validate(args.report, args.expected_sources, args.repository_root)
        if args.json_output:
            args.json_output.write_text(json.dumps(summary, indent=2) + "\n")
    except (ValueError, OSError, sqlite3.Error, TypeError) as error:
        print(f"Mull report validation failed: {error}", file=sys.stderr)
        return 1
    labels = [*EVALUATED_STATUSES.values(), "not_covered"]
    print("Validated Mull report: " + ", ".join(f"{label}={summary[label]}" for label in labels))
    score = summary["mutation_score"]
    print(f"Mutation score: {score['percent']:.2f}% ({summary['killed']}/{score['denominator']}); {score['definition']}")
    coverage = summary["evaluation_coverage"]
    print(f"Evaluated / discovered: {coverage['evaluated']}/{coverage['discovered']} ({coverage['percent']:.2f}%); diagnostic only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
