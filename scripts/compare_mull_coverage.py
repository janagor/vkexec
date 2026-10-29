"""Compare Mull automatic/raw/indexed coverage against one unchanged executable.

Run from the clang-mut shell. All commands and full debug logs are retained.
The output directory is dedicated to this experiment; named outputs are replaced.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess


def verify_fixture(database):
    with sqlite3.connect(database) as connection:
        fixture_rows = connection.execute(
            "SELECT mutant_id, execution_status, line_number, filename FROM mutant"
        ).fetchall()
    uncalled = [(identifier, status) for identifier, status, line, filename in fixture_rows
                if Path(filename).name == "ranges.hpp" and line == 4]
    called = [(identifier, status) for identifier, status, line, filename in fixture_rows
              if not (Path(filename).name == "ranges.hpp" and line == 4)]
    if (len({identifier for identifier, _, _, _ in fixture_rows}) != 10
            or len(uncalled) != 2 or any(status != 8 for _, status in uncalled)
            or len(called) != 8 or any(status not in (1, 2) for _, status in called)):
        raise RuntimeError("Indexed fixture regression: expected eight killed/survived called mutants "
                           "and two uncovered mutants in ranges.hpp:4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("test_args", nargs="*")
    parser.add_argument("--verify-fixture", action="store_true",
                        help="Require eight called mutants evaluated and two uncalled mutants uncovered in indexed mode")
    args = parser.parse_args()
    binary, config, report = (p.resolve() for p in (args.binary, args.config, args.report_dir))
    report.mkdir(parents=True, exist_ok=True)
    commands = []

    def run(command, log, env=None):
        effective_env = os.environ if env is None else env
        profile_env = [f"{key}={effective_env[key]}" for key in ("LLVM_PROFILE_FILE", "MULL_CONFIG")
                       if key in effective_env]
        unset_env = [arg for key in ("LLVM_PROFILE_FILE", "MULL_CONFIG")
                     if key not in effective_env for arg in ("-u", key)]
        commands.append(shlex.join(["env", *unset_env, *profile_env, *command]))
        (report / "commands.txt").write_text("\n".join(commands) + "\n")
        with (report / log).open("w") as output:
            subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)

    raw, indexed = report / "manual.profraw", report / "manual.profdata"
    for path in (raw, indexed):
        path.unlink(missing_ok=True)
    run([str(binary), *args.test_args], "original.log", dict(os.environ, LLVM_PROFILE_FILE=str(raw)))
    run(["llvm-profdata", "merge", "-sparse", str(raw), "-o", str(indexed)], "merge.log")
    for path, label in ((raw, "raw"), (indexed, "indexed")):
        run(["llvm-profdata", "show", "--all-functions", "--counts", str(path)], f"{label}-counts.txt")
    library_args = []
    for directory in os.environ.get("MULL_LIBRARY_PATH", "").split(":"):
        if directory:
            library_args += ["--ld-search-path", directory]
    result = {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "modes": {}}
    for mode, profile in (("automatic", None), ("raw", raw), ("indexed", indexed)):
        database = report / f"{mode}.sqlite"
        database.unlink(missing_ok=True)  # Mull appends to existing reports.
        command = [os.environ.get("MULL_RUNNER", "mull-runner-22"), "--allow-surviving",
                   "--include-not-covered", "--timeout", "10000", "--debug", "--debug-coverage",
                   *library_args, "--reporters", "SQLite", "--report-dir", str(report),
                   "--report-name", mode]
        if profile:
            command += ["--coverage-info", str(profile)]
        run([*command, str(binary), *args.test_args], f"{mode}.log", dict(os.environ, MULL_CONFIG=str(config)))
        with sqlite3.connect(database) as connection:
            rows = connection.execute("SELECT mutant_id, execution_status FROM mutant ORDER BY mutant_id").fetchall()
        result["modes"][mode] = {"counts": dict(Counter(status for _, status in rows)), "mutants": rows}
    sets = [{identifier for identifier, _ in data["mutants"]} for data in result["modes"].values()]
    if not sets[0] == sets[1] == sets[2]:
        raise RuntimeError("Discovered mutant IDs differ between profile modes")
    if hashlib.sha256(binary.read_bytes()).hexdigest() != result["binary_sha256"]:
        raise RuntimeError("Executable changed during the comparison")
    (report / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.verify_fixture:
        verify_fixture(report / "indexed.sqlite")
        print("Indexed fixture regression check passed (8 evaluated, 2 genuinely uncovered)")
    print(json.dumps({mode: data["counts"] for mode, data in result["modes"].items()}))


if __name__ == "__main__":
    main()
