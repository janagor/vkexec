"""Configure-only scope checks; no Mull, Vulkan, or mutation compilation."""
from pathlib import Path
import subprocess
import os
import json
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
MODULE = HERE.parents[2] / "cmake/MutationTesting.cmake"


class RegistrationTests(unittest.TestCase):
    def test_execution_and_failures(self):
        with tempfile.TemporaryDirectory(prefix="mutation registration ") as directory:
            root = Path(directory)
            # Use the real pipeline and validator with a miniature repository root.
            source = root / "source"
            shutil.copytree(HERE, source)
            (source / "scripts").mkdir()
            shutil.copyfile(MODULE.parent.parent / "scripts/validate_mull_report.py",
                            source / "scripts/validate_mull_report.py")
            build = root / "build"
            result = subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
                                     f"-DMUTATION_MODULE={MODULE}", "-DCASE=integration"], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for stage in ("", "configure", "baseline", "merge", "runner", "validation"):
                log = root / "args.log"
                log.unlink(missing_ok=True)
                env = dict(os.environ, AUDIT_LOG=str(log), AUDIT_SOURCE=str(source / "dummy.cpp"), FAIL_STAGE=stage)
                result = subprocess.run(["cmake", "--build", str(build), "--target", "vkexec_mutation_ranges"],
                                        env=env, capture_output=True)
                with self.subTest(stage=stage):
                    self.assertEqual(result.returncode == 0, not stage, result.stdout + result.stderr)
                    if stage == "configure":
                        self.assertFalse(log.exists())
                        continue
                    lines = log.read_text().splitlines()
                    self.assertEqual(lines[0], "baseline|[ranges]|--order|lex|--abort|argument with spaces")
                    if not stage:
                        runner_entry = json.loads(lines[2])
                        runner = runner_entry[1]
                        self.assertEqual(runner[-5:], ["[ranges]", "--order", "lex", "--abort", "argument with spaces"])
                        self.assertEqual(Path(runner_entry[2]), build / "mutation-builds/ranges/mutation-configs/ranges.yml")
                    self.assertEqual(len(lines), {"baseline": 1, "merge": 2}.get(stage, 3))

            # Restore a valid ranges report after the intentionally invalid report.
            env = dict(os.environ, AUDIT_LOG=str(log), AUDIT_SOURCE=str(source / "dummy.cpp"), FAIL_STAGE="")
            result = subprocess.run(["cmake", "--build", str(build), "--target", "vkexec_mutation_ranges"],
                                    env=env, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # Both public targets reuse their own persistent tree and preserve filters.
            log.unlink(missing_ok=True)
            env = dict(os.environ, AUDIT_LOG=str(log), AUDIT_SOURCE=str(source / "dummy.cpp"), FAIL_STAGE="")
            result = subprocess.run(["cmake", "--build", str(build), "--target", "vkexec_mutation_other_uncovered"],
                                    env=env, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            lines = log.read_text().splitlines()
            self.assertEqual(lines[0], "baseline|other-argument")
            runner = json.loads(lines[2])
            self.assertEqual(runner[1][-1], "other-argument")
            self.assertIn("--include-not-covered", runner[1])
            self.assertEqual(Path(runner[2]), build / "mutation-builds/other/mutation-configs/other.yml")
            for name in ("ranges", "other"):
                self.assertTrue((build / "mutation-reports" / f"vkexec_core_{name}{'_uncovered' if name == 'other' else ''}.json").exists())

    def test_registration(self):
        with tempfile.TemporaryDirectory(prefix="mutation registration ") as directory:
            inherited_value = 'spaces;quote";close]==];literal${unexpanded}'
            for case in ("valid", "missing_config", "missing_target", "empty_sources",
                         "duplicate", "invalid_name", "relative_config"):
                build = Path(directory) / case
                result = subprocess.run(["cmake", "-S", str(HERE), "-B", str(build), "-G", "Ninja",
                                         f"-DMUTATION_MODULE={MODULE}", f"-DCASE={case}",
                                         f"-Dvkexec_AUDIT_TEST_VALUE={inherited_value}"],
                                        capture_output=True, text=True)
                with self.subTest(case=case):
                    if case not in ("valid", "relative_config"):
                        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                        continue
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    rules = (build / "build.ninja").read_text()
                    for name, config in (("ranges", "scope.yml"), ("other", "inactive.yml")):
                        self.assertIn(f"vkexec_mutation_{name}:", rules)
                        self.assertIn(f"vkexec_mutation_{name}_uncovered:", rules)
                        self.assertEqual((build / f"mutation-configs/{name}.yml").read_text(),
                                         (HERE / config).read_text())
                    for target in ("vkexec", "dummy"):
                        self.assertNotIn("-fpass-plugin=", (build / f"{target}-options.txt").read_text())
                    for name in ("ranges", "other"):
                        internal = build / "mutation-builds" / name
                        result = subprocess.run(["cmake", "-S", str(HERE), "-B", str(internal), "-G", "Ninja",
                                                 "-C", str(build / "mutation-configs/cache.cmake"),
                                                 f"-DMUTATION_MODULE={MODULE}", f"-DCASE={case}",
                                                 f"-Dvkexec_MULL_INTERNAL_SCOPE={name}"], capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertEqual((internal / "inherited-value.txt").read_text(), inherited_value)
                        child_rules = (internal / "build.ninja").read_text()
                        self.assertIn(f"vkexec_mutation_{name}:", child_rules)
                        self.assertIn("mutation-configs/" + name + ".yml", child_rules)
                        if name == "ranges":
                            self.assertEqual(child_rules.count('"argument with spaces"'), 4)
                            self.assertNotIn("other-argument", child_rules)
                        else:
                            self.assertEqual(child_rules.count("other-argument"), 4)
                            self.assertNotIn('"argument with spaces"', child_rules)
                        for target in ("vkexec", "dummy"):
                            self.assertEqual((internal / f"{target}-options.txt").read_text().count("-fpass-plugin="), 1)
                        self.assertEqual((internal / "mull.yml").read_text(),
                                         (build / f"mutation-configs/{name}.yml").read_text())


if __name__ == "__main__":
    unittest.main()
