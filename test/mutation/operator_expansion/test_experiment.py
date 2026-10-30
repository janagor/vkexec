"""Orchestrator checks using Git fixtures and synthetic SQLite reports, no Mull."""
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import mutation_operator_experiment as e


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'repo'
        self.root.mkdir()
        for command in (['init', '-q'], ['config', 'user.email', 'fixture@example.com'],
                        ['config', 'user.name', 'Fixture']):
            self.git(*command)
        for source in e.sources('resource_helpers'):
            path = self.root / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('// fixture\n')
        config = self.root / e.CONFIG
        config.parent.mkdir(parents=True)
        config.write_text('canonical\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture')
        self.artifacts = Path(self.temp.name) / 'evidence'

    def git(self, *args):
        return e.git(self.root, *args)

    def row(self, root=None, index=0, status='killed', op=None):
        file = e.sources('resource_helpers')[index]
        op = op or e.NEW[index]
        return {'file': file, 'line': 1, 'column': 2, 'mutator': op, 'status': status,
                'mutant_id': f'{op}:{root or self.root}/{file}:1:2:1:3'}

    def test_configurations(self):
        for variant in ('or', 'combined'):
            text = e.configuration(variant)
            e.validate_configuration(text, variant)
            for bad in (text.replace(e.NEW[0], 'unknown'), text + 'excludePaths: []\n',
                        text.replace('resource_state_tracker_helpers', 'other')):
                with self.assertRaises(ValueError):
                    e.validate_configuration(bad, variant)

    def test_exact_id_normalization(self):
        row = self.row()
        self.assertEqual(e.normalize(row, self.root), row['mutant_id'].replace(str(self.root) + '/', ''))
        for path in ('/other/checkout', str(self.root) + '-suffix'):
            with self.assertRaises(ValueError):
                e.normalize(self.row(Path(path)), self.root)

    def reports(self, directory, root, duplicate=False, bad_source=False, bad_status=False):
        directory.mkdir(parents=True)
        for mode in ('covered', 'inclusive'):
            db = directory / (mode + '.sqlite')
            with closing(sqlite3.connect(db)) as conn:
                conn.execute('CREATE TABLE mutant (filename, directory, execution_status, mutator, line_number, column_number, mutant_id)')
                rows = [self.row(root, i) for i in range(2)]
                if duplicate:
                    rows.append(rows[0])
                for row in rows:
                    conn.execute('INSERT INTO mutant VALUES (?, ?, ?, ?, ?, ?, ?)',
                                 ('evil.hpp' if bad_source else row['file'], str(root), 99 if bad_status else 1,
                                  row['mutator'], 1, 2, row['mutant_id']))
                conn.commit()
            if not bad_source and not bad_status:
                summary = e.parse_report(db, [root / p for p in e.sources('resource_helpers')], root)
                (directory / (mode + '.json')).write_text(json.dumps(summary))

    def test_reports_independently_validated(self):
        self.reports(self.artifacts, self.root)
        self.assertEqual(len(e.load_pair(self.artifacts, self.root)), 2)
        data = json.loads((self.artifacts / 'covered.json').read_text())
        data['locations'][0]['status'] = 'survived'
        (self.artifacts / 'covered.json').write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'SQLite/JSON'):
            e.load_pair(self.artifacts, self.root)

    def test_covered_can_omit_only_uncovered_source(self):
        self.reports(self.artifacts, self.root)
        helper = e.sources('resource_helpers')[0]
        with closing(sqlite3.connect(self.artifacts / 'covered.sqlite')) as db:
            db.execute('DELETE FROM mutant WHERE filename = ?', (helper,))
            db.commit()
        (self.artifacts / 'covered.json').unlink()
        with closing(sqlite3.connect(self.artifacts / 'inclusive.sqlite')) as db:
            db.execute('UPDATE mutant SET execution_status = 8 WHERE filename = ?', (helper,))
            db.commit()
        summary = e.parse_report(self.artifacts / 'inclusive.sqlite',
                                 [self.root / p for p in e.sources('resource_helpers')], self.root)
        (self.artifacts / 'inclusive.json').write_text(json.dumps(summary))
        rows = e.load_pair(self.artifacts, self.root)
        self.assertEqual(len(rows), 2)
        self.assertTrue((self.artifacts / 'validated-covered.json').is_file())
        with closing(sqlite3.connect(self.artifacts / 'inclusive.sqlite')) as db:
            db.execute('UPDATE mutant SET execution_status = 1 WHERE filename = ?', (helper,))
            db.commit()
        summary = e.parse_report(self.artifacts / 'inclusive.sqlite',
                                 [self.root / p for p in e.sources('resource_helpers')], self.root)
        (self.artifacts / 'inclusive.json').write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'contradiction'):
            e.load_pair(self.artifacts, self.root)

    def test_json_command_output_remains_parseable(self):
        path = self.artifacts.with_suffix('.json')
        e.execute([sys.executable, '-c', 'print("{}")'], self.root, path)
        self.assertEqual(e.read(path), {})
        self.assertTrue(path.with_suffix('.command.txt').is_file())

    def test_duplicate_ids(self):
        self.reports(self.artifacts, self.root, duplicate=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            e.load_pair(self.artifacts, self.root)

    def test_missing_reports(self):
        with self.assertRaises(ValueError):
            e.load_pair(self.artifacts, self.root)

    def test_invalid_source_and_status(self):
        for option in ('bad_source', 'bad_status'):
            directory = self.artifacts / option
            self.reports(directory, self.root, **{option: True})
            with self.assertRaises(ValueError):
                e.load_pair(directory, self.root)

    def test_classifications_preserve_status(self):
        for classification in e.CLASSIFICATIONS:
            self.assertEqual(e.classify(self.row(status='not_covered'), classification)['status'], 'not_covered')
        with self.assertRaises(ValueError):
            e.classify(self.row(), 'equivalent so killed')

    def test_cross_operator_comparison(self):
        before = [{'id': f'old:{i}', 'status': 'killed'} for i in range(56)]
        rows = {r['id']: {**r, 'mutator': 'cxx_eq_to_ne'} for r in before}
        rows['new'] = {'status': 'survived', 'mutator': e.NEW[0]}
        self.assertTrue(e.compatibility(rows, {'mutants': before})['compatible'])
        del rows['old:0']
        rows['old:1']['status'] = 'survived'
        result = e.compatibility(rows, {'mutants': before})
        self.assertFalse(result['compatible'])
        self.assertEqual(len(result['changes']), 2)

    def test_repeat_and_combined_new_subset(self):
        row = self.row()
        row['id'] = e.normalize(row, self.root)
        row.pop('mutant_id')
        row = e.classify(row)
        first = dict(failed=False, revision='abc', toolchain_versions={}, test_args=[],
                     wrapper_sha256='a' * 64, mutants=[row])
        second = {**first, 'mutants': [row, {**row, 'id': row['id'].replace(e.NEW[0], 'cxx_eq_to_ne'), 'mutator': 'cxx_eq_to_ne'}]}
        result = e.compare_experiments([first, second])
        self.assertTrue(result['stable'])
        self.assertFalse(result['wrapper_changed'])
        different_wrapper = {**second, 'wrapper_sha256': 'b' * 64}
        result = e.compare_experiments([first, different_wrapper])
        self.assertTrue(result['stable'])
        self.assertTrue(result['wrapper_changed'])
        changed = {**first, 'mutants': [{**row, 'status': 'survived'}]}
        self.assertFalse(e.compare_experiments([first, changed])['stable'])
        with self.assertRaises(ValueError):
            e.compare_experiments([first, {**first, 'mutants': [row, row]}])

    def test_overlay_cleanup_and_original_preservation_on_failure(self):
        seen = []
        def command(args, cwd, log):
            if args[0] == sys.executable:
                seen.append(cwd)
                self.assertEqual((cwd / e.CONFIG).read_text(), e.configuration('or'))
                self.assertEqual(e.git(cwd, 'diff', '--name-only'), e.CONFIG)
                log.write_text('simulated build failure')
                raise ValueError('simulated failure')
            e.execute(args, cwd, log)
        with self.assertRaisesRegex(ValueError, 'simulated'):
            e.experiment(self.root, 'or', 'HEAD', self.artifacts, command=command,
                         verify=lambda p: [], versions=lambda: {})
        self.assertEqual(len(seen), 1)
        self.assertFalse(seen[0].exists())
        self.assertEqual((self.root / e.CONFIG).read_text(), 'canonical\n')
        self.assertEqual(self.git('status', '--porcelain'), '')
        self.assertEqual(len(self.git('worktree', 'list', '--porcelain').split('worktree ')), 2)
        self.assertTrue((self.artifacts / 'audit.log').is_file())
        self.assertTrue(json.loads((self.artifacts / 'experiment.json').read_text())['failed'])

    def test_successful_simulated_audit(self):
        def command(args, cwd, log):
            if args[0] == 'git':
                return e.execute(args, cwd, log)
            log.write_text('simulated command output')
            if 'scripts/ci/run_mutation_audits.py' in args:
                destination = self.artifacts / 'audit'
                self.reports(destination / 'resource_helpers', cwd)
                manifest = dict(revision=e.git(cwd, 'rev-parse', 'HEAD'), source_clean=False,
                                failed=False, execution={})
                (destination / 'manifest.json').write_text(json.dumps(manifest))
                binary = cwd / 'out/build/unixlike-clang-mut/mutation-builds/resource_helpers/test/vkexec_core_tests'
                binary.parent.mkdir(parents=True)
                binary.write_bytes(b'fixture binary')
                # Match the production checkout's ignored build directory.
            elif 'scripts/compare_mull_coverage.py' in args:
                (self.artifacts / 'coverage').mkdir()
                (self.artifacts / 'coverage/indexed.log').write_text('debug coverage')
            else:
                log.write_text('{}')
        diagnostic = {'locations': [{**self.row(index=i), 'mutant_id': e.normalize(self.row(index=i), self.root),
                                    'classification': 'source_covered', 'instances': []} for i in range(2)]}
        # Ignore synthetic build outputs in the fixture, as the real repo does.
        (self.root / '.gitignore').write_text('out/\n')
        self.git('add', '.gitignore')
        self.git('commit', '-qm', 'ignore build')
        with patch.object(e, 'fingerprint', return_value={}), patch.object(e, 'diagnose', return_value=diagnostic):
            result = e.experiment(self.root, 'or', 'HEAD', self.artifacts, command=command,
                                  verify=lambda p: [], versions=lambda: {})
        self.assertFalse(result['failed'])
        self.assertEqual(len(result['mutants']), 2)
        self.assertTrue(result['binary_sha256'])
        self.assertEqual(self.git('status', '--porcelain'), '')

    def test_silently_ignored_operator_rejected(self):
        directory = self.artifacts
        def command(args, cwd, log, env=None):
            if '--reporters' in args:
                with closing(sqlite3.connect(directory / 'operators.sqlite')) as conn:
                    conn.execute('CREATE TABLE mutant (mutator, execution_status)')
                    conn.execute('INSERT INTO mutant VALUES (?, 1)', (e.NEW[0],))
                    conn.commit()
        with patch.object(e, 'execute', side_effect=command), patch.dict(os.environ, {'MULL_PLUGIN': 'fake-plugin'}):
            with self.assertRaisesRegex(ValueError, 'not emitted'):
                e.verify_operators(directory)

    def test_dirty_revision_rejected(self):
        (self.root / e.CONFIG).write_text('dirty')
        with self.assertRaisesRegex(ValueError, 'clean committed'):
            e.experiment(self.root, 'or', 'HEAD', self.artifacts)
        self.assertFalse(self.artifacts.exists())

    def test_unexpected_overlay_rejected(self):
        (self.root / e.CONFIG).write_text('overlay')
        e.check_overlay(self.root)
        (self.root / 'untracked').write_text('oops')
        with self.assertRaises(ValueError):
            e.check_overlay(self.root)


if __name__ == '__main__':
    unittest.main()
