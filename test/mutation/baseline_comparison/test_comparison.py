"""Comparator tests require only the Python standard library."""
from collections import Counter
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import compare_mutation_baselines as c


def counts(rows):
    values = Counter(r['status'] for r in rows)
    return {'total': len(rows), **{s: values[s] for s in c.STATUSES}}


def report(rows):
    result = {'schema_version': 2, **counts(rows), 'locations': copy.deepcopy(rows)}
    for field, key in [('per_file', 'file'), ('per_mutator', 'mutator')]:
        result[field] = {group: counts([r for r in rows if r[key] == group]) for group in {r[key] for r in rows}}
    return result


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.scope = 'dag'
        self.rows = json.loads((Path(__file__).parent / 'fixtures/dag.json').read_text())
        execution = {'toolchain': {'clang': '22.1.8', 'llvm': '22.1.8', 'mull': '0.35.0'}, 'build_options': {'CMAKE_BUILD_TYPE': 'Debug'}, 'timeout_policy': {'minimum_ms': 1000, 'warmup_ms': 10000, 'workers': 1}}
        self.fp = c.fingerprint(ROOT, self.scope, execution)
        self.baseline = {'schema_version': 1, 'scope': self.scope, 'baseline_revision': 'a' * 40, 'fingerprint': copy.deepcopy(self.fp), 'mutants': [{**{k: v for k, v in row.items() if k != 'mutant_id'}, 'id': row['mutant_id']} for row in self.rows]}

    def compare(self, policy='regression'):
        inclusive = report(self.rows)
        covered = report([r for r in self.rows if r['status'] != 'not_covered'])
        return c.compare(self.scope, self.baseline, covered, inclusive, ROOT, self.fp, policy)

    def test_identical_and_known_outcomes(self):
        self.assertEqual(self.compare()['exit_code'], 0)
        self.assertTrue(self.compare()['comparable'])

    def test_empty_and_missing_fields(self):
        with self.assertRaises(ValueError):
            c.report_map(report([]), self.scope, ROOT)
        malformed = report(self.rows)
        del malformed['per_mutator']
        with self.assertRaises(ValueError):
            c.report_map(malformed, self.scope, ROOT)
        del self.baseline['fingerprint']['timeout_policy']
        with self.assertRaises(ValueError):
            self.compare()

    def test_absent_file(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(OSError):
            c.read(Path(directory) / 'missing.json')

    def test_checkout_prefix(self):
        for root in ('/tmp/checkout-one', '/tmp/checkout-two'):
            rows = copy.deepcopy(self.rows)
            for row in rows:
                row['mutant_id'] = row['mutant_id'].replace('cxx_eq_to_ne:', f'cxx_eq_to_ne:{root}/')
            self.assertEqual(set(c.rows_map(rows, self.scope, root)), set(c.rows_map(self.rows, self.scope, ROOT)))
        with self.assertRaises(ValueError):
            c.rows_map(rows, self.scope, '/tmp/other')

    def test_regressions(self):
        for status in c.STATUSES[1:]:
            with self.subTest(status=status):
                self.rows[0]['status'] = status
                result = self.compare()
                self.assertEqual(result['exit_code'], 2)
                self.assertEqual(result['regressions'], 1)
                self.assertEqual(self.compare('report')['exit_code'], 0)

    def test_survivor_becomes_uncovered(self):
        self.rows[1]['status'] = 'not_covered'
        result = self.compare()
        self.assertTrue(result['comparable'])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['exit_code'], 2)
        self.assertEqual(result['regressions'], 1)
        self.assertEqual(self.compare('report')['exit_code'], 0)

    def test_registration_change_still_compares(self):
        self.fp['test_registration_sha256']['test/CMakeLists.txt'] = 'b' * 64
        self.assertEqual(self.compare()['exit_code'], 0)
        self.rows[0]['status'] = 'survived'
        result = self.compare()
        self.assertTrue(result['comparable'])
        self.assertTrue(result['test_registration_changed'])
        self.assertEqual(result['exit_code'], 2)

    def test_legacy_registration_fingerprint(self):
        fp = self.baseline['fingerprint']
        registration = fp.pop('test_registration_sha256')
        fp['infrastructure_sha256'].update(registration)
        self.assertEqual(self.compare()['exit_code'], 0)
        self.fp['test_registration_sha256']['test/CMakeLists.txt'] = 'b' * 64
        self.rows[0]['status'] = 'survived'
        self.assertTrue(self.compare()['comparable'])
        self.assertEqual(self.compare()['exit_code'], 2)

    def test_instrumentation_change_still_requires_review(self):
        self.fp['infrastructure_sha256']['cmake/MutationTesting.cmake'] = 'b' * 64
        self.rows[0]['status'] = 'survived'
        result = self.compare()
        self.assertFalse(result['comparable'])
        self.assertEqual(result['regressions'], 1)
        self.assertEqual(result['exit_code'], 3)

    def test_new_execution_instability(self):
        self.rows[1]['status'] = 'abnormal_exit'
        self.assertEqual(self.compare()['exit_code'], 2)

    def test_improvements(self):
        for row in self.rows:
            row['status'] = 'killed'
        self.assertEqual(self.compare()['exit_code'], 0)
        self.assertEqual(sum(r['classification'] == 'improvement' for r in self.compare()['changes']), 2)

    def test_population(self):
        self.rows.pop()
        self.assertEqual(self.compare()['exit_code'], 3)
        self.rows = [{**{k: v for k, v in row.items() if k != 'id'}, 'mutant_id': row['id']} for row in self.baseline['mutants']]
        new = copy.deepcopy(self.rows[0])
        new['line'] = 25
        new['mutant_id'] = new['mutant_id'].replace(':22:20:22:', ':25:20:25:')
        self.rows.append(new)
        self.assertEqual(self.compare()['exit_code'], 3)

    def test_duplicate_normalized_id(self):
        row = copy.deepcopy(self.rows[0])
        row['mutant_id'] = row['mutant_id'].replace('cxx_eq_to_ne:', f'cxx_eq_to_ne:{ROOT}/')
        self.rows.append(row)
        with self.assertRaises(ValueError):
            self.compare()

    def test_malformed(self):
        for field, value in [('status', 'unknown'), ('file', 'include/other.hpp'), ('line', 0), ('mutator', '')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                original = copy.deepcopy(self.rows)
                self.rows[0][field] = value
                try:
                    self.compare()
                finally:
                    self.rows = original

    def test_schema_and_counts(self):
        for field, value in [('schema_version', 1), ('total', 999), ('killed', True)]:
            malformed = report(self.rows)
            malformed[field] = value
            with self.assertRaises(ValueError):
                c.report_map(malformed, self.scope, ROOT)
        self.baseline['schema_version'] = 2
        with self.assertRaises(ValueError):
            self.compare()

    def test_summary_groups(self):
        malformed = report(self.rows)
        malformed['per_file'][self.rows[0]['file']]['killed'] = 999
        with self.assertRaises(ValueError):
            c.report_map(malformed, self.scope, ROOT)

    def test_contradiction(self):
        covered = copy.deepcopy(self.rows[:2])
        covered[0]['status'] = 'survived'
        with self.assertRaises(ValueError):
            c.validate_pair(report(covered), report(self.rows), self.scope, ROOT)

    def test_fingerprints_require_review(self):
        for field, value in [('source_sha256', {c.sources('dag')[0]: 'b' * 64}), ('operators', ['cxx_boundary']), ('toolchain', {'clang': '23', 'llvm': '23', 'mull': '0.36'})]:
            original = copy.deepcopy(self.fp)
            self.fp[field] = value
            self.assertEqual(self.compare()['exit_code'], 3)
            self.fp = original

    def test_assertion_weakening_still_compares(self):
        self.fp['test_source_sha256']['test/core/dag_planning_tests.cpp'] = 'b' * 64
        self.rows[0]['status'] = 'survived'
        self.assertTrue(self.compare()['tests_changed'])
        self.assertEqual(self.compare()['exit_code'], 2)

    def test_scope_isolation(self):
        self.rows[0]['status'] = 'survived'
        dag = self.compare()
        rows = [{**{k: v for k, v in row.items() if k != 'id'}, 'mutant_id': row['id']} for row in self.baseline['mutants']]
        for row in rows:
            row['file'] = c.sources('ranges')[0]
            row['mutant_id'] = row['mutant_id'].replace(c.sources('dag')[0], row['file'])
        fp = copy.deepcopy(self.fp)
        fp['source_sha256'] = {c.sources('ranges')[0]: 'a' * 64}
        fp['test_args'][0] = '[ranges]'
        baseline = {**self.baseline, 'scope': 'ranges', 'fingerprint': fp, 'mutants': [{**{k: v for k, v in row.items() if k != 'mutant_id'}, 'id': row['mutant_id']} for row in rows]}
        for row in rows:
            row['status'] = 'killed'
        ranges = c.compare('ranges', baseline, report(rows), report(rows), ROOT, fp)
        self.assertEqual(dag['exit_code'], 2)
        self.assertEqual(ranges['exit_code'], 0)
        with self.assertRaises(ValueError):
            c.compare('ranges', self.baseline, report(rows), report(rows), ROOT, fp)

    def test_injected_regression_cli_exit(self):
        self.rows[0]['status'] = 'survived'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            values = {
                'baseline': self.baseline,
                'covered': report(self.rows[:2]),
                'inclusive': report(self.rows),
                'execution-metadata': {key: self.fp[key] for key in ('toolchain', 'build_options', 'timeout_policy')},
            }
            command = [sys.executable, str(ROOT / 'scripts/compare_mutation_baselines.py'), '--scope', 'dag', '--repository-root', str(ROOT), '--output', str(path / 'result.json')]
            for flag, value in values.items():
                file = path / (flag + '.json')
                file.write_text(json.dumps(value))
                command.extend(['--' + flag, str(file)])
            for policy, code in [('report', 0), ('regression', 2)]:
                result = subprocess.run([*command, '--policy', policy], capture_output=True)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(json.loads((path / 'result.json').read_text())['regressions'], 1)

    def test_invalid_cli_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            metadata = path / 'bad.json'
            for contents in ('{', '{}'):
                metadata.write_text(contents)
                command = [sys.executable, str(ROOT / 'scripts/compare_mutation_baselines.py'), '--scope', 'dag', '--repository-root', str(ROOT), '--output', str(path / 'result.json'), '--execution-metadata', str(metadata)]
                for flag in ('--baseline', '--covered', '--inclusive'):
                    command.extend([flag, str(path / 'absent.json')])
                result = subprocess.run(command, capture_output=True)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(json.loads((path / 'result.json').read_text())['status'], 'invalid')


if __name__ == '__main__':
    unittest.main()
