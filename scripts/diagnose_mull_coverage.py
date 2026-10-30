"""Compare Mull locations with llvm-cov export source regions, retaining instantiations.

Source execution does not prove that Mull's selected binary instance executed.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from validate_mull_report import parse_report


def diagnose(summary: dict, coverage: dict, repository_root: Path, mull_log: str | None = None) -> dict:
    if coverage.get("type") != "llvm.coverage.json.export":
        raise ValueError("Expected llvm-cov export JSON")
    functions = [(function, [Path(name).resolve() for name in function["filenames"]])
                 for data in coverage["data"] for function in data.get("functions", [])]
    uncovered = []
    if mull_log is not None:
        for filename, *coordinates in re.findall(r"mull-coverage: (.*):(\d+):(\d+):(\d+):(\d+)$", mull_log, re.MULTILINE):
            uncovered.append((Path(filename).resolve(), list(map(int, coordinates))))
    locations = []
    for location in summary["locations"]:
        source = (repository_root / location["file"]).resolve()
        point = (location["line"], location["column"])
        instances = []
        for function, filenames in functions:
            if source in filenames:
                regions = []
                for region in function["regions"]:
                    # LLVM code regions only; skip expansion, skipped, gap and branch regions.
                    start_line, start_col, end_line, end_col, count, file_id, _, kind = region
                    if (kind == 0 and filenames[file_id] == source
                            and (start_line, start_col) <= point < (end_line, end_col)):
                        regions.append(region)
                if regions:
                    # The innermost region describes the expression, rather than its enclosing function.
                    region = max(regions, key=lambda r: (r[0], r[1], -r[2], -r[3]))
                    instances.append({"function": function["name"], "count": region[4],
                                      "region": region, "function_count": function.get("count"),
                                      "containing_gap_regions": [r for r in function["regions"]
                                          if r[7] == 3 and filenames[r[5]] == source
                                          and (r[0], r[1]) <= point < (r[2], r[3])]})
        executed = any(instance["count"] > 0 for instance in instances)
        classification = (
            "source_covered_attribution_discrepancy" if executed and location["status"] == "not_covered"
            else "source_covered" if executed
            else "source_region_not_executed" if instances
            else "no_source_region"
        )
        locations.append({**location, "classification": classification,
                          "instances": sorted(instances, key=lambda item: (item["function"], item["count"]))})
        if mull_log is not None:
            # Mull 0.35 uses inclusive end coordinates, unlike LLVM code regions.
            locations[-1]["mull_uncovered_ranges"] = sorted({tuple(r) for filename, r in uncovered
                if filename == source and tuple(r[:2]) <= point <= tuple(r[2:])})
    return {"schema_version": 1,
            "classification_counts": dict(sorted(Counter(row["classification"] for row in locations).items())),
            "locations": locations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("source", type=Path, nargs="+", help="Complete report source allowlist")
    parser.add_argument("coverage_export", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--mull-log", type=Path, help="Debug log with --debug --debug-coverage")
    args = parser.parse_args()
    result = diagnose(parse_report(args.report, args.source, args.repository_root),
                      json.loads(args.coverage_export.read_text()), args.repository_root,
                      args.mull_log.read_text() if args.mull_log else None)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["classification_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
