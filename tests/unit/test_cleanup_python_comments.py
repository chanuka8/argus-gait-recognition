import ast
import tempfile
import unittest
from pathlib import Path

from scripts.cleanup_python_comments import (
    analyze_python_file,
    discover_python_files,
    run_apply,
    run_audit,
    run_check,
    run_dry_run,
    transform_source,
)


class TestCleanupPythonCommentsUnit(unittest.TestCase):
    def test_01_module_docstring_removal(self):
        source = '"""Module documentation."""\nx = 1\n'
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            self.assertEqual(len(res.docstrings), 1)
            self.assertEqual(res.docstrings[0].node_type, "module")
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertNotIn("Module documentation", transformed)
            self.assertIn("x = 1", transformed)

    def test_02_class_docstring_removal(self):
        source = "class Target:\n    \"\"\"Class documentation.\"\"\"\n    val = 10\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            self.assertEqual(len(res.docstrings), 1)
            self.assertEqual(res.docstrings[0].node_type, "class")
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertNotIn("Class documentation", transformed)
            self.assertIn("val = 10", transformed)

    def test_03_function_docstring_removal(self):
        source = "def compute():\n    \"\"\"Function docstring.\"\"\"\n    return 42\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            self.assertEqual(len(res.docstrings), 1)
            self.assertEqual(res.docstrings[0].node_type, "function")
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertNotIn("Function docstring", transformed)
            self.assertIn("return 42", transformed)

    def test_04_async_function_docstring_removal(self):
        source = "async def fetch():\n    \"\"\"Async function docstring.\"\"\"\n    return await get()\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            self.assertEqual(len(res.docstrings), 1)
            self.assertEqual(res.docstrings[0].node_type, "async_function")
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertNotIn("Async function docstring", transformed)
            self.assertIn("return await get()", transformed)

    def test_05_triple_single_quote_docstring_removal(self):
        source = "def action():\n    '''Triple single quote docstring.'''\n    return True\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            self.assertEqual(len(res.docstrings), 1)
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertNotIn("Triple single quote docstring", transformed)
            self.assertIn("return True", transformed)

    def test_06_runtime_multiline_string_preservation(self):
        source = 'msg = """\nLine 1\nLine 2\nLine 3\n"""\n'
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            self.assertEqual(len(res.docstrings), 0)
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertEqual(transformed, source)

    def test_07_multiline_sql_string_preservation(self):
        source = 'sql_query = """\nSELECT id, name\nFROM gallery\nWHERE score >= 0.70\n"""\n'
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            self.assertEqual(len(res.docstrings), 0)
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertEqual(transformed, source)

    def test_08_runtime_prompt_template_preservation(self):
        source = 'prompt = """\nYou are an AI assistant.\n# Instruction: Analyze gait silhouettes.\n"""\n'
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            self.assertEqual(len(res.docstrings), 0)
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertIn("# Instruction: Analyze gait silhouettes.", transformed)

    def test_09_ordinary_full_line_comment_removal(self):
        source = "# Temporary calculation note\nx = 10\n# Another note\ny = 20\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            self.assertEqual(len(res.removable_comments), 2)
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertNotIn("Temporary calculation note", transformed)
            self.assertNotIn("Another note", transformed)
            self.assertIn("x = 10", transformed)
            self.assertIn("y = 20", transformed)

    def test_10_inline_comment_removal(self):
        source = "alpha = 0.35  # Exponential moving average filter coefficient\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            self.assertEqual(len(res.removable_comments), 1)
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertNotIn("Exponential moving average", transformed)
            self.assertEqual(transformed.strip(), "alpha = 0.35")

    def test_11_hash_inside_normal_string_preservation(self):
        source = 'color_hex = "#FF0000"\ntag = "Item #123"\n'
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            self.assertEqual(len(res.comments), 0)
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertEqual(transformed, source)

    def test_12_tooling_directives_preservation(self):
        source = (
            "import os  # noqa: F401\n"
            "from typing import Any  # type: ignore\n"
            "# pragma: no cover\n"
            "# pylint: disable=unused-variable\n"
            "# fmt: off\n"
            "# isort: skip\n"
            "def fn():\n"
            "    # nosec\n"
            "    pass\n"
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            self.assertEqual(len(res.removable_comments), 0)
            self.assertGreaterEqual(len(res.preserved_comments), 6)
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertIn("noqa: F401", transformed)
            self.assertIn("type: ignore", transformed)
            self.assertIn("pragma: no cover", transformed)
            self.assertIn("pylint: disable=unused-variable", transformed)

    def test_13_shebang_preservation(self):
        source = "#!/usr/bin/env python\nx = 1\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            self.assertEqual(len(res.removable_comments), 0)
            self.assertTrue(any("shebang" in (c.preservation_reason or "").lower() for c in res.preserved_comments))
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertIn("#!/usr/bin/env python", transformed)

    def test_14_encoding_declaration_preservation(self):
        source = "# -*- coding: utf-8 -*-\nx = 1\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            self.assertEqual(len(res.removable_comments), 0)
            self.assertTrue(any("encoding" in (c.preservation_reason or "").lower() for c in res.preserved_comments))
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertIn("# -*- coding: utf-8 -*-", transformed)

    def test_15_empty_function_and_class_pass_insertion(self):
        source = "class EmptyClass:\n    \"\"\"Doc only.\"\"\"\n\ndef empty_func():\n    \"\"\"Doc only.\"\"\"\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            tree = ast.parse(transformed)
            compile(transformed, "mod.py", "exec")
            cls_node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
            func_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
            self.assertTrue(isinstance(cls_node.body[0], ast.Pass))
            self.assertTrue(isinstance(func_node.body[0], ast.Pass))

    def test_16_crlf_preservation(self):
        source_crlf = "# ordinary comment\r\nx = 1\r\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source_crlf.encode("utf-8"))
            res = analyze_python_file(p)
            self.assertEqual(res.line_ending, "\r\n")
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertIn("\r\n", transformed)
            self.assertNotIn("\r\r\n", transformed)

    def test_17_lf_preservation(self):
        source_lf = "# ordinary comment\nx = 1\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source_lf.encode("utf-8"))
            res = analyze_python_file(p)
            self.assertEqual(res.line_ending, "\n")
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertNotIn("\r\n", transformed)
            self.assertIn("\n", transformed)

    def test_18_idempotency(self):
        source = "def foo():\n    \"\"\"Doc.\"\"\"\n    # comment\n    return 1\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res1 = analyze_python_file(p)
            t1, err1 = transform_source(res1)
            self.assertIsNone(err1)
            p.write_bytes(t1.encode("utf-8"))
            res2 = analyze_python_file(p)
            t2, err2 = transform_source(res2)
            self.assertIsNone(err2)
            self.assertEqual(t1, t2)

    def test_19_syntax_validation_after_transformation(self):
        source = "def valid_func(a, b):\n    \"\"\"Docstring.\"\"\"\n    # comment\n    return a + b\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            compiled = compile(transformed, "<test>", "exec")
            self.assertIsNotNone(compiled)

    def test_20_dry_run_does_not_modify_files(self):
        source = "# Comment\nx = 10\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            run_dry_run([p])
            self.assertEqual(p.read_bytes().decode("utf-8"), source)

    def test_21_check_mode_returns_expected_status(self):
        with tempfile.TemporaryDirectory() as td:
            p_dirty = Path(td) / "dirty.py"
            p_dirty.write_bytes(b"# comment\nx = 1\n")
            p_clean = Path(td) / "clean.py"
            p_clean.write_bytes(b"x = 1\n")

            self.assertEqual(run_check([p_dirty]), 1)
            self.assertEqual(run_check([p_clean]), 0)

    def test_22_self_exclusion(self):
        files = discover_python_files(Path("scripts"), include_self=False)
        self.assertTrue(all(f.name != "cleanup_python_comments.py" for f in files))
        files_with_self = discover_python_files(Path("scripts"), include_self=True)
        self.assertTrue(any(f.name == "cleanup_python_comments.py" for f in files_with_self))

    def test_23_excluded_directory_handling(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            venv_dir = root / ".venv" / "lib"
            venv_dir.mkdir(parents=True)
            (venv_dir / "ignored.py").write_bytes(b"# note\nx = 1\n")
            (root / "valid.py").write_bytes(b"# note\nx = 1\n")

            discovered = discover_python_files(root)
            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0].name, "valid.py")

    def test_24_syntax_error_file_handling(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "invalid.py"
            p.write_bytes(b"def broken(\n")
            res = analyze_python_file(p)
            self.assertIsNotNone(res.parse_error)
            run_audit([p])
            run_dry_run([p])
            run_apply([p])
            self.assertEqual(p.read_bytes(), b"def broken(\n")

    def test_25_mixed_comments_docstrings_runtime_strings(self):
        source = (
            "#!/usr/bin/env python\n"
            '"""Module doc."""\n'
            "import os  # noqa: F401\n"
            "# removable comment\n"
            'PROMPT = """\n'
            "Analyze gait #123\n"
            '"""\n'
            "class Pipeline:\n"
            '    """Class doc."""\n'
            "    def run(self):\n"
            '        """Method doc."""\n'
            "        # intermediate comment\n"
            '        query = "SELECT * FROM gallery"\n'
            "        return query\n"
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)
            transformed, err = transform_source(res)
            self.assertIsNone(err)
            self.assertIn("#!/usr/bin/env python", transformed)
            self.assertIn("noqa: F401", transformed)
            self.assertIn("Analyze gait #123", transformed)
            self.assertIn('query = "SELECT * FROM gallery"', transformed)
            self.assertNotIn("Module doc.", transformed)
            self.assertNotIn("Class doc.", transformed)
            self.assertNotIn("Method doc.", transformed)
            self.assertNotIn("removable comment", transformed)

    def test_26_backup_option_in_apply(self):
        source = "# removable\nx = 100\n"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "script.py"
            p.write_bytes(source.encode("utf-8"))
            backup_dir = root / "backups"
            run_apply([p], backup_dir=backup_dir)
            self.assertTrue((backup_dir / "script.py").exists())
            self.assertEqual((backup_dir / "script.py").read_bytes().decode("utf-8"), source)
            self.assertNotIn("removable", p.read_bytes().decode("utf-8"))

    def test_27_no_remove_flags(self):
        source = '"""Doc."""\n# comment\nx = 1\n'
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.py"
            p.write_bytes(source.encode("utf-8"))
            res = analyze_python_file(p)

            # Preserve docstrings
            t_no_docs, _ = transform_source(res, remove_docstrings=False, remove_comments=True)
            self.assertIn("Doc.", t_no_docs)
            self.assertNotIn("comment", t_no_docs)

            # Preserve comments
            t_no_comms, _ = transform_source(res, remove_docstrings=True, remove_comments=False)
            self.assertNotIn("Doc.", t_no_comms)
            self.assertIn("comment", t_no_comms)


if __name__ == "__main__":
    unittest.main()
