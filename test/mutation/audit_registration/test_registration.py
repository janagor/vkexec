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
        with tempfile.TemporaryDirectory() as directory:
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
            for stage in ("", "baseline", "merge", "runner", "validation"):
                log = root / "args.log"
                log.unlink(missing_ok=True)
                env = dict(os.environ, AUDIT_LOG=str(log), AUDIT_SOURCE=str(source / "dummy.cpp"), FAIL_STAGE=stage)
                result = subprocess.run(["cmake", "--build", str(build), "--target", "vkexec_mutation_ranges"],
                                        env=env, capture_output=True)
                with self.subTest(stage=stage):
                    self.assertEqual(result.returncode == 0, not stage, result.stdout + result.stderr)
                    lines = log.read_text().splitlines()
                    self.assertEqual(lines[0], "baseline|[ranges]|--order|lex|--abort|argument with spaces")
                    if not stage:
                        runner = json.loads(lines[2])[1]
                        self.assertEqual(runner[-5:], ["[ranges]", "--order", "lex", "--abort", "argument with spaces"])
                    self.assertEqual(len(lines), {"baseline": 1, "merge": 2}.get(stage, 3))

    def test_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            rules = {}
            for case in ("valid", "unknown", "missing_config", "missing_target", "empty_sources",
                         "duplicate", "invalid_name", "inactive", "active_other", "relative_config"):
                build = Path(directory) / case
                active = "other" if case in ("active_other", "relative_config") else "ranges"
                result = subprocess.run(["cmake", "-S", str(HERE), "-B", str(build), "-G", "Ninja",
                                         f"-DMUTATION_MODULE={MODULE}", f"-DCASE={case}",
                                         "-Dvkexec_MULL_SCOPE=" + ("absent" if case == "unknown" else active)],
                                        capture_output=True, text=True)
                with self.subTest(case=case):
                    if case not in ("valid", "inactive", "active_other", "relative_config"):
                        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                        continue
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    rules[case] = (build / "build.ninja").read_text()
                    self.assertIn(f"vkexec_mutation_{active}:", rules[case])
                    self.assertIn(f"vkexec_mutation_{active}_uncovered:", rules[case])
                    if active == "other":
                        self.assertNotIn("vkexec_mutation_ranges", rules[case])
                        self.assertNotIn('"argument with spaces"', rules[case])
                        self.assertEqual(rules[case].count("inactive-argument"), 4)
                    else:
                        self.assertEqual(rules[case].count('"argument with spaces"'), 4)
                        self.assertNotIn("inactive-argument", rules[case])
                        self.assertNotIn("vkexec_mutation_other", rules[case])
                    for target in ("vkexec", "dummy"):
                        self.assertEqual((build / f"{target}-options.txt").read_text().count("-fpass-plugin="), 1)
                    config = "inactive.yml" if active == "other" else "scope.yml"
                    self.assertEqual((build / "mull.yml").read_text(), (HERE / config).read_text())


if __name__ == "__main__":
    unittest.main()
