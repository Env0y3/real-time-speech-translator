"""Run the offline suite and write explicit outcome counts (no service calls)."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    result_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "test-results.json"
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    planned = suite.countTestCases()
    result = None
    try:
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        if result is not None:
            imports = sum(isinstance(test, unittest.loader._FailedTest) for test, _ in result.errors)
            data = {
                "planned": planned, "executed": result.testsRun,
                "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
                "assertion_failures": len(result.failures),
                "module_import_errors": imports,
                "other_errors": len(result.errors) - imports,
                "skipped": len(result.skipped),
                "not_run": max(0, planned - result.testsRun),
                "complete": result.testsRun == planned,
                "skipped_tests": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
            }
        else:
            data = {"planned": planned, "complete": False, "interrupted": True,
                    "note": "Runner interrupted; consult per-test console output, do not count all as passed."}
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result is not None and result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
