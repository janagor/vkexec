"""Checks for precise region selection and independent inline instantiations."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import diagnose_mull_coverage
from diagnose_mull_coverage import diagnose
from validate_mull_report import parse_report, validate


class CoverageTests(unittest.TestCase):
    def test_uncovered_only_cli_preserves_distinct_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "range.hpp"
            source.touch()
            report = root / "audit.sqlite"
            coverage = root / "coverage.json"
            output = root / "diagnosis.json"
            with closing(sqlite3.connect(report)) as db, db:
                db.execute("CREATE TABLE mutant (filename TEXT, directory TEXT, execution_status INT, "
                           "mutator TEXT, line_number INT, column_number INT, mutant_id TEXT)")
                db.executemany("INSERT INTO mutant VALUES (?, '', 8, 'comparison', 10, 20, ?)",
                               [(str(source), identifier) for identifier in ['instance-b', 'instance-a']])
            summary = parse_report(report, source, root)
            self.assertIsNone(summary['mutation_score']['percent'])
            self.assertEqual(summary['evaluation_coverage']['percent'], 0)
            with self.assertRaises(ValueError):
                validate(report, source, root)
            coverage.write_text(json.dumps({"type": "llvm.coverage.json.export", "data": [{
                "functions": [{"name": "inline_instance", "filenames": [str(source)],
                               "regions": [[10, 1, 11, 1, 7, 0, 0, 0]]}]}]}))
            result = subprocess.run([sys.executable, diagnose_mull_coverage.__file__, str(report),
                                     str(source), str(coverage), '--repository-root', str(root),
                                     '--json-output', str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            diagnosis = json.loads(output.read_text())
            self.assertEqual(diagnosis['classification_counts'], {'source_covered_attribution_discrepancy': 2})
            self.assertEqual([row['mutant_id'] for row in diagnosis['locations']], ['instance-a', 'instance-b'])

    def test_regions_and_instantiations(self):
        root = Path('/tmp')
        location = {"file": "range.hpp", "line": 10, "column": 20,
                    "mutator": "comparison", "status": "not_covered"}

        def function(name, count):
            return {"name": name, "filenames": ['/tmp/range.hpp'], "regions": [
                [1, 1, 20, 1, 100, 0, 0, 0],
                [10, 10, 10, 30, count, 0, 0, 0],
                [10, 15, 10, 25, 0, 0, 0, 3],  # Gap must not hide a code region.
            ]}

        def compare(functions, row=location):
            return diagnose({"locations": [row]}, {
                "type": "llvm.coverage.json.export", "data": [{"functions": functions}]
            }, root)["locations"][0]

        self.assertEqual(compare([function('a', 0)])["classification"], 'source_region_not_executed')
        result = compare([function('a', 0), function('b', 7)])
        self.assertEqual(result['classification'], 'source_covered_attribution_discrepancy')
        self.assertEqual(result['instances'], [{'function': 'a', 'count': 0}, {'function': 'b', 'count': 7}])
        self.assertEqual(compare([function('a', 7)], {**location, 'status': 'killed'})['classification'], 'source_covered')
        self.assertEqual(compare([])['classification'], 'no_source_region')
        # End positions are exclusive.
        narrow = function('a', 7)
        narrow['regions'] = [[10, 10, 10, 30, 7, 0, 0, 0]]
        self.assertEqual(compare([narrow], {**location, 'column': 29})['classification'], 'source_covered_attribution_discrepancy')
        self.assertEqual(compare([narrow], {**location, 'column': 30})['classification'], 'no_source_region')


if __name__ == '__main__':
    unittest.main()
