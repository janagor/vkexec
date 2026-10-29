"""Behavioral checks for rejecting ineffective Mull reports."""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

import validate_mull_report


class ReportTests(unittest.TestCase):
    def test_report_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "resource_ranges.hpp"
            source.touch()
            report = root / "audit.sqlite"
            with self.assertRaises(ValueError):
                validate_mull_report.validate(report, source)
            self.assertFalse(report.exists())
            with closing(sqlite3.connect(report)) as connection, connection:
                connection.execute("CREATE TABLE mutant (filename TEXT, directory TEXT, execution_status INT)")
            cases = (
                ("empty", [], False),
                ("evaluated", [(str(source), "", 1), (str(source), "", 2)], True),
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
                        connection.executemany("INSERT INTO mutant VALUES (?, ?, ?)", rows)
                    if valid:
                        self.assertTrue(validate_mull_report.validate(report, source))
                    else:
                        with self.assertRaises(ValueError):
                            validate_mull_report.validate(report, source)
            with closing(sqlite3.connect(report)) as connection, connection:
                connection.execute("DROP TABLE mutant")
            with self.assertRaises(sqlite3.Error):
                validate_mull_report.validate(report, source)


if __name__ == "__main__":
    unittest.main()
