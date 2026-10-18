"""Collection regressions run even on machines without Qt/audio packages."""
import builtins
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch


class GuiCollectionTests(unittest.TestCase):
    def _load_with_import_error(self, error):
        original_import = builtins.__import__
        def checked_import(name, *args, **kwargs):
            if name == "PySide6":
                raise error
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=checked_import):
            return runpy.run_path(str(Path(__file__).with_name("test_gui.py")))

    def test_missing_pyside_collects_skipped_class_without_annotation_error(self):
        namespace = self._load_with_import_error(
            ModuleNotFoundError("No module named PySide6", name="PySide6")
        )
        self.assertFalse(namespace["PYSIDE_AVAILABLE"])
        self.assertTrue(namespace["MainWindowTests"].__unittest_skip__)

    def test_unrelated_missing_module_is_not_hidden(self):
        with self.assertRaises(ModuleNotFoundError):
            self._load_with_import_error(
                ModuleNotFoundError("Missing nested dependency", name="unrelated")
            )

    def test_broken_qt_import_is_not_hidden(self):
        with self.assertRaises(ImportError):
            self._load_with_import_error(ImportError("Qt shared library failure"))
