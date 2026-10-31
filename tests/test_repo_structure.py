import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import repo_check


class TestRepositoryStructure(unittest.TestCase):
    def test_required_paths_exist(self):
        self.assertEqual(repo_check.check_required_paths(), [])

    def test_documentation_consistency(self):
        failures = repo_check.run_checks()
        self.assertEqual(failures, [])

    def test_four_gb_constraint_documented(self):
        self.assertTrue(repo_check.check_text_contains("README.md", "4 GB"))
        self.assertTrue(repo_check.check_text_contains("docs/TECHNICAL_CONSTRAINTS.md", "Maximum 4 GB RAM per training process"))


if __name__ == "__main__":
    unittest.main()
