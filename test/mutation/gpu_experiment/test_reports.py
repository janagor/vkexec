from pathlib import Path
import sqlite3
from contextlib import closing
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import gpu_mutation_experiment as gpu


class Reports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / gpu.SOURCE).parent.mkdir(parents=True)
        (self.root / gpu.SOURCE).touch()
        for mode in ('covered', 'inclusive'):
            d = self.root / mode
            d.mkdir()
            for suffix in ('profraw', 'profdata'):
                (d / ('profile.' + suffix)).write_bytes(b'profile')
            with closing(sqlite3.connect(d / 'report.sqlite')) as db:
                db.execute('CREATE TABLE mutant(filename, directory, execution_status, mutator, line_number, column_number, mutant_id)')
                db.execute('INSERT INTO mutant VALUES (?, ?, ?, ?, ?, ?, ?)', self.row())
                db.commit()

    def row(self, status=1, source=gpu.SOURCE):
        return (source, str(self.root), status, 'cxx_lt_to_le', 1, 2,
                f'cxx_lt_to_le:{source}:1:2:1:3')

    def change(self, sql, args=()):
        with closing(sqlite3.connect(self.root / 'inclusive/report.sqlite')) as db:
            db.execute(sql, args)
            db.commit()

    def test_valid_pair_and_intersection(self):
        rows = gpu.load_reports(self.root, self.root)
        cpu = {'mutants': [dict(next(iter(rows.values())), status='survived')]}
        result = gpu.compare_cpu(rows, cpu)
        self.assertEqual(result['intersection'], 1)
        self.assertIn('requires assertion attribution', result['outcomes'][0]['classification'])
        self.assertIsNone(result['mutation_score'])

    def test_duplicate(self):
        self.change('INSERT INTO mutant VALUES (?, ?, ?, ?, ?, ?, ?)', self.row())
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            gpu.load_reports(self.root, self.root)

    def test_unknown_source(self):
        self.change('UPDATE mutant SET filename=?', ('other.hpp',))
        with self.assertRaisesRegex(ValueError, 'Unexpected mutated source'):
            gpu.load_reports(self.root, self.root)

    def test_unknown_status(self):
        self.change('UPDATE mutant SET execution_status=7')
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            gpu.load_reports(self.root, self.root)

    def test_inconsistent_status(self):
        self.change('UPDATE mutant SET execution_status=2')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            gpu.load_reports(self.root, self.root)

    def test_missing_profile(self):
        (self.root / 'covered/profile.profdata').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing profile'):
            gpu.load_reports(self.root, self.root)

    def test_invalid_sqlite(self):
        (self.root / 'covered/report.sqlite').write_text('invalid')
        with self.assertRaises(sqlite3.DatabaseError):
            gpu.load_reports(self.root, self.root)

    def test_empty_covered(self):
        with closing(sqlite3.connect(self.root / 'covered/report.sqlite')) as db:
            db.execute('DELETE FROM mutant')
            db.commit()
        self.change('UPDATE mutant SET execution_status=8')
        self.assertEqual(next(iter(gpu.load_reports(self.root, self.root).values()))['status'], 'not_covered')

    def test_baseline_skips_and_empty(self):
        p = self.root / 'test.xml'
        for content in ('<testsuite skipped="1"><testcase/><testcase/></testsuite>', '<testsuites/>', '<testsuites><testcase/><testcase><skipped/></testcase></testsuites>'):
            p.write_text(content)
            with self.assertRaises(ValueError):
                gpu.validate_baseline(p)
        p.write_text('<testsuites><testcase/><testcase/></testsuites>')
        gpu.validate_baseline(p)

    def test_empty_covered_validator_exception_is_narrow(self):
        import subprocess
        d = self.root / 'covered'
        (d / 'mull.log').write_text('Mull report validation failed: Mull report contains no mutants')
        error = subprocess.CalledProcessError(1, ['cmake'])
        self.assertFalse(gpu.empty_covered_validator_failure(error, d))
        with closing(sqlite3.connect(d / 'report.sqlite')) as db:
            db.execute('DELETE FROM mutant')
            db.commit()
        self.assertTrue(gpu.empty_covered_validator_failure(error, d))
        self.assertFalse(gpu.empty_covered_validator_failure(subprocess.TimeoutExpired(['cmake'], 1), d))

    def test_json_contradiction(self):
        (self.root / 'inclusive/report.json').write_text('{"locations": []}')
        with self.assertRaisesRegex(ValueError, 'SQLite/JSON contradiction'):
            gpu.load_reports(self.root, self.root)
