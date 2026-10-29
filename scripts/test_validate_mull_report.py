"""Behavioral checks for rejecting ineffective Mull reports."""

from contextlib import closing
from pathlib import Path
import sqlite3
import json
import subprocess
import sys
import tempfile
import unittest

import validate_mull_report


class ReportTests(unittest.TestCase):
    def test_summary_and_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "resource_ranges.hpp"
            source.touch()
            report = root / "audit.sqlite"
            output = root / "summary.json"
            with closing(sqlite3.connect(report)) as connection, connection:
                connection.execute("CREATE TABLE mutant (filename TEXT, directory TEXT, execution_status INT, mutator TEXT, line_number INT DEFAULT 1, column_number INT DEFAULT 1, mutant_id TEXT DEFAULT 'fixture-id')")
                connection.executemany(
                    "INSERT INTO mutant (filename, directory, execution_status, mutator) VALUES (?, ?, ?, ?)",
                    [(str(source), "", status, mutator) for status, mutator in
                     [(1, "comparison"), (1, "comparison"), (2, "boundary"),
                      (3, "boundary"), (4, "boundary"), (5, "boundary"), (8, "uncovered")]],
                )
            summary = validate_mull_report.validate(report, source, root)
            self.assertEqual(summary["total"], 7)
            self.assertEqual(summary["schema_version"], 2)
            self.assertEqual(summary["evaluation_coverage"]["evaluated"], 6)
            self.assertAlmostEqual(summary["evaluation_coverage"]["percent"], 600 / 7)
            self.assertEqual(len(summary["locations"]), 7)
            self.assertEqual(summary["locations"][0]["file"], source.name)
            self.assertEqual([summary[key] for key in
                              ["killed", "survived", "timed_out", "crashed", "abnormal_exit", "not_covered"]],
                             [2, 1, 1, 1, 1, 1])
            self.assertEqual(summary["mutation_score"]["denominator"], 3)
            self.assertAlmostEqual(summary["mutation_score"]["percent"], 200 / 3)
            self.assertEqual(summary["per_file"][source.name]["total"], 7)
            self.assertEqual(summary["per_mutator"]["comparison"]["mutation_score"]["percent"], 100)
            self.assertEqual(summary["per_mutator"]["boundary"]["mutation_score"]["percent"], 0)
            self.assertIsNone(summary["per_mutator"]["uncovered"]["mutation_score"]["percent"])
            # Identical reports in another checkout must produce identical JSON.
            with tempfile.TemporaryDirectory() as other_directory:
                other_root = Path(other_directory)
                other_source = other_root / source.name
                other_source.touch()
                other_report = other_root / report.name
                with closing(sqlite3.connect(report)) as original, closing(sqlite3.connect(other_report)) as relocated:
                    original.backup(relocated)
                    relocated.execute("UPDATE mutant SET filename = ?", (str(other_source),))
                    relocated.commit()
                self.assertEqual(validate_mull_report.validate(other_report, other_source, other_root), summary)
            (root / "outside").mkdir()
            with self.assertRaises(ValueError):
                validate_mull_report.validate(report, source, root / "outside")
            result = subprocess.run(
                [sys.executable, validate_mull_report.__file__, str(report), str(source),
                 "--json-output", str(output), "--repository-root", str(root)], capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text()), summary)
            self.assertIn("killed=2", result.stdout)
            self.assertIn("66.67% (2/3)", result.stdout)

    def test_multiple_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [root / name for name in ("a.hpp", "b.cpp", "c.cpp")]
            for source in sources:
                source.touch()
            report = root / "audit.sqlite"
            with closing(sqlite3.connect(report)) as db, db:
                db.execute("CREATE TABLE mutant (filename TEXT, directory TEXT, execution_status INT, mutator TEXT, line_number INT, column_number INT, mutant_id TEXT)")
                db.executemany("INSERT INTO mutant VALUES (?, '', 1, 'comparison', 1, 1, 'id')", [(str(p),) for p in sources[:2]])
            summary = validate_mull_report.validate(report, sources[:2], root)
            self.assertEqual(summary["killed"], 2)
            for status in (2, 8):
                with closing(sqlite3.connect(report)) as db, db:
                    db.execute("UPDATE mutant SET execution_status=? WHERE filename=?", (status, str(sources[1])))
                summary = validate_mull_report.validate(report, sources[:2], root)
                self.assertEqual(summary["per_file"]["b.cpp"]["total"], 1)
                self.assertEqual(summary["mutation_score"]["percent"], 50 if status == 2 else 100)
            (root / "sub").mkdir()
            for expected in ([], sources, [sources[0]], [sources[0], root / "sub" / ".." / "a.hpp"]):
                with self.subTest(expected=expected), self.assertRaises(ValueError):
                    validate_mull_report.validate(report, expected, root)
            result = subprocess.run([sys.executable, validate_mull_report.__file__, str(report),
                                     *map(str, sources[:2]), "--repository-root", str(root)], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for invalid in ("", "   ", None, b"invalid"):
                with closing(sqlite3.connect(report)) as db, db:
                    db.execute("UPDATE mutant SET mutant_id=?", (invalid,))
                with self.assertRaises(ValueError):
                    validate_mull_report.parse_report(report, sources[:2], root)

    def test_report_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "resource_ranges.hpp"
            source.touch()
            report = root / "audit.sqlite"
            with self.assertRaises(ValueError):
                validate_mull_report.validate(report, source, root)
            self.assertFalse(report.exists())
            with closing(sqlite3.connect(report)) as connection, connection:
                connection.execute("CREATE TABLE mutant (filename TEXT, directory TEXT, execution_status INT, mutator TEXT DEFAULT 'cxx_test', line_number INT DEFAULT 1, column_number INT DEFAULT 1, mutant_id TEXT DEFAULT 'fixture-id')")
            cases = (
                ("empty", [], False),
                ("evaluated", [(str(source), "", 1), (str(source), "", 2)], True),
                ("timeouts only", [(str(source), "", 3)], False),
                ("crashes only", [(str(source), "", 4), (str(source), "", 5)], False),
                ("uncovered only", [(str(source), "", 8)], False),
                ("mixed coverage", [(str(source), "", 2), (str(source), "", 8)], True),
                ("wrong source", [(str(root / "other.hpp"), "", 2)], False),
                ("dry run", [(str(source), "", 6)], False),
                ("unknown status", [(str(source), "", 99)], False),
                ("relative source", [(source.name, str(root), 2)], True),
                ("ambiguous source", [(source.name, "", 2)], False),
            )
            for name, rows, valid in cases:
                with self.subTest(name=name):
                    with closing(sqlite3.connect(report)) as connection, connection:
                        connection.execute("DELETE FROM mutant")
                        connection.executemany("INSERT INTO mutant (filename, directory, execution_status) VALUES (?, ?, ?)", rows)
                    if valid:
                        self.assertTrue(validate_mull_report.validate(report, source, root))
                    else:
                        with self.assertRaises(ValueError):
                            validate_mull_report.validate(report, source, root)
                    if name == "empty":
                        with self.assertRaisesRegex(ValueError, "Mull report contains no mutants"):
                            validate_mull_report.parse_report(report, source, root)
                    elif name == "uncovered only":
                        summary = validate_mull_report.parse_report(report, source, root)
                        self.assertEqual(summary["not_covered"], 1)
                        self.assertIsNone(summary["mutation_score"]["percent"])
            with closing(sqlite3.connect(report)) as connection, connection:
                connection.execute("DROP TABLE mutant")
            with self.assertRaises(sqlite3.Error):
                validate_mull_report.validate(report, source, root)


if __name__ == "__main__":
    unittest.main()
