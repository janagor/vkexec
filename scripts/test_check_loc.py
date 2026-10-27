"""Tests for the C++ file length check."""

import subprocess
import tempfile
import unittest
from pathlib import Path

import check_loc


class LimitTests(unittest.TestCase):
    def test_limits_and_boundaries(self) -> None:
        cases = (
            ("src/main.cpp", check_loc.SOURCE_LIMIT),
            ("include/main.hpp", check_loc.HEADER_LIMIT),
            ("test/core/main.cpp", check_loc.TEST_LIMIT),
            ("benchmark/main.hpp", check_loc.TEST_LIMIT),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            for name, limit in cases:
                with self.subTest(name=name):
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.assertEqual(check_loc.limit_for(Path(name)), limit)
                    path.write_bytes(b"x\n" * limit)
                    rows = check_loc.check(root)
                    row = next(row for row in rows if row[0] == Path(name))
                    self.assertLessEqual(row[1], row[2])
                    path.write_bytes(b"x\n" * (limit + 1))
                    rows = check_loc.check(root)
                    row = next(row for row in rows if row[0] == Path(name))
                    self.assertGreater(row[1], row[2])

    def test_excluded_paths(self) -> None:
        for name in ("out/main.cpp", "third_party/main.hpp", "generated/main.cpp", "src/main.txt"):
            with self.subTest(name=name):
                self.assertIsNone(check_loc.limit_for(Path(name)))

    def test_physical_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.cpp"
            for content, expected in ((b"", 0), (b"a\r\nb\r\n", 2), (b"a\nb", 2)):
                with self.subTest(content=content):
                    path.write_bytes(content)
                    self.assertEqual(check_loc.count_lines(path), expected)

    def test_git_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text("ignored.cpp\n", encoding="utf-8")
            (root / "tracked.cpp").write_bytes(b"x\n" * (check_loc.SOURCE_LIMIT + 1))
            (root / "new.hpp").write_bytes(b"x\n" * (check_loc.HEADER_LIMIT + 1))
            (root / "ignored.cpp").write_bytes(b"x\n" * (check_loc.SOURCE_LIMIT + 1))
            (root / "vendor").mkdir()
            (root / "vendor" / "third_party.cpp").write_bytes(b"x\n" * (check_loc.SOURCE_LIMIT + 1))
            subprocess.run(["git", "-C", str(root), "add", "tracked.cpp"], check=True)

            rows = check_loc.check(root)
            by_name = {str(path): (lines, limit) for path, lines, limit in rows}
            self.assertEqual(set(by_name), {"tracked.cpp", "new.hpp"})
            self.assertEqual(
                by_name["tracked.cpp"],
                (check_loc.SOURCE_LIMIT + 1, check_loc.SOURCE_LIMIT),
            )
            self.assertEqual(
                by_name["new.hpp"],
                (check_loc.HEADER_LIMIT + 1, check_loc.HEADER_LIMIT),
            )


if __name__ == "__main__":
    unittest.main()
