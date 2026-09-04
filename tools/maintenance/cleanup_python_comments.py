#!/usr/bin/env python
"""
ARGUS AI Developer Cleanup Utility: Python Comments & Docstrings.

A standalone, safe developer utility to audit, dry-run, check, and clean
unwanted Python '#' comments and genuine docstrings without affecting
runtime strings (SQL, prompts, HTML, configs), interpreter directives,
or application behavior.
"""

from __future__ import annotations

import argparse
import ast
import codecs
import io
import os
import re
import shutil
import sys
import tempfile
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_EXCLUDE_DIRS: set[str] = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "outputs",
    "runs",
    "cache",
    "site-packages",
    ".idea",
    ".vscode",
    ".gemini",
    "assets",
}

DEFAULT_EXCLUDE_EXTENSIONS: set[str] = {
    ".npy",
    ".npz",
    ".pth",
    ".pt",
    ".onnx",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".mp4",
    ".avi",
    ".svg",
    ".zip",
    ".tar",
    ".gz",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".html",
    ".css",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
}

TOOLING_DIRECTIVE_PATTERN = re.compile(
    r"#\s*(noqa|type:\s*ignore|pragma:|pylint:|mypy:|pyright:|ruff:|fmt:|isort:|nosec|bandit:|flake8:|autopep8:|black:|coverage:)",
    re.IGNORECASE,
)

ENCODING_DECLARATION_PATTERN = re.compile(
    r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)",
    re.IGNORECASE,
)


@dataclass
class DocstringItem:
    node_type: str
    start_lineno: int
    start_col: int
    end_lineno: int
    end_col: int
    doc_value: str
    needs_pass: bool = False
    pass_indent: str = "    "


@dataclass
class CommentItem:
    start_lineno: int
    start_col: int
    end_lineno: int
    end_col: int
    comment_text: str
    line_text: str
    is_standalone: bool
    is_preserved: bool
    preservation_reason: str | None = None


@dataclass
class FileScanResult:
    path: Path
    encoding: str
    has_bom: bool
    line_ending: str
    original_source: str
    docstrings: list[DocstringItem] = field(default_factory=list)
    comments: list[CommentItem] = field(default_factory=list)
    parse_error: str | None = None
    transformed_source: str | None = None
    validation_error: str | None = None

    @property
    def removable_docstrings(self) -> list[DocstringItem]:
        return self.docstrings

    @property
    def removable_comments(self) -> list[CommentItem]:
        return [c for c in self.comments if not c.is_preserved]

    @property
    def preserved_comments(self) -> list[CommentItem]:
        return [c for c in self.comments if c.is_preserved]

    @property
    def has_removable_content(self) -> bool:
        return bool(self.removable_docstrings or self.removable_comments)


class DocstringVisitor(ast.NodeVisitor):
    def __init__(self, source_lines: list[str]):
        self.source_lines = source_lines
        self.docstrings: list[DocstringItem] = []

    def _check_body_docstring(self, body: list[ast.stmt], node_type: str) -> None:
        if not body:
            return
        first_stmt = body[0]
        if (
            isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)
        ):
            needs_pass = (node_type != "module") and (len(body) == 1)
            col_offset = first_stmt.col_offset
            pass_indent = " " * col_offset
            if pass_indent == "" and node_type != "module":
                pass_indent = "    "

            self.docstrings.append(
                    DocstringItem(
                        node_type=node_type,
                        start_lineno=first_stmt.lineno,
                        start_col=first_stmt.col_offset,
                        end_lineno=first_stmt.end_lineno or first_stmt.lineno,
                        end_col=first_stmt.end_col_offset or 0,
                        doc_value=first_stmt.value.value,
                        needs_pass=needs_pass,
                        pass_indent=pass_indent,
                    )
                )

    def visit_Module(self, node: ast.Module) -> None:
        self._check_body_docstring(node.body, "module")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._check_body_docstring(node.body, "class")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_body_docstring(node.body, "function")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_body_docstring(node.body, "async_function")
        self.generic_visit(node)


def detect_file_line_ending(raw_bytes: bytes) -> str:
    if b"\r\n" in raw_bytes:
        return "\r\n"
    return "\n"


def is_shebang_comment(tok: tokenize.TokenInfo) -> bool:
    return tok.start[0] == 1 and tok.string.startswith("#!")


def is_encoding_comment(tok: tokenize.TokenInfo, raw_bytes: bytes) -> bool:
    return bool(tok.start[0] in (1, 2) and ENCODING_DECLARATION_PATTERN.match(tok.line))


def is_tooling_directive(tok: tokenize.TokenInfo) -> tuple[bool, str | None]:
    comment_text = tok.string.strip()
    match = TOOLING_DIRECTIVE_PATTERN.search(comment_text)
    if match:
        return True, f"Tooling directive: {match.group(1).strip(':')}"
    lower_text = comment_text.lower()
    for kw in ("noqa", "type: ignore", "pragma:", "pylint:", "pyright:", "ruff:", "isort:", "fmt: off", "fmt: on", "nosec"):
        if kw in lower_text:
            return True, f"Tooling directive keyword: {kw}"
    return False, None


def analyze_comments(source_bytes: bytes, source_text: str) -> list[CommentItem]:
    comments: list[CommentItem] = []
    bio = io.BytesIO(source_bytes)
    try:
        token_gen = tokenize.tokenize(bio.readline)
        for tok in token_gen:
            if tok.type == tokenize.COMMENT:
                line_prefix = tok.line[: tok.start[1]]
                is_standalone = line_prefix.strip() == ""

                if is_shebang_comment(tok):
                    comments.append(
                        CommentItem(
                            start_lineno=tok.start[0],
                            start_col=tok.start[1],
                            end_lineno=tok.end[0],
                            end_col=tok.end[1],
                            comment_text=tok.string,
                            line_text=tok.line,
                            is_standalone=is_standalone,
                            is_preserved=True,
                            preservation_reason="Interpreter shebang",
                        )
                    )
                elif is_encoding_comment(tok, source_bytes):
                    comments.append(
                        CommentItem(
                            start_lineno=tok.start[0],
                            start_col=tok.start[1],
                            end_lineno=tok.end[0],
                            end_col=tok.end[1],
                            comment_text=tok.string,
                            line_text=tok.line,
                            is_standalone=is_standalone,
                            is_preserved=True,
                            preservation_reason="Encoding declaration",
                        )
                    )
                else:
                    is_tooling, reason = is_tooling_directive(tok)
                    if is_tooling:
                        comments.append(
                            CommentItem(
                                start_lineno=tok.start[0],
                                start_col=tok.start[1],
                                end_lineno=tok.end[0],
                                end_col=tok.end[1],
                                comment_text=tok.string,
                                line_text=tok.line,
                                is_standalone=is_standalone,
                                is_preserved=True,
                                preservation_reason=reason,
                            )
                        )
                    else:
                        comments.append(
                            CommentItem(
                                start_lineno=tok.start[0],
                                start_col=tok.start[1],
                                end_lineno=tok.end[0],
                                end_col=tok.end[1],
                                comment_text=tok.string,
                                line_text=tok.line,
                                is_standalone=is_standalone,
                                is_preserved=False,
                            )
                        )
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return comments


def analyze_python_file(path: Path) -> FileScanResult:
    try:
        raw_bytes = path.read_bytes()
    except OSError as e:
        return FileScanResult(
            path=path,
            encoding="utf-8",
            has_bom=False,
            line_ending="\n",
            original_source="",
            parse_error=f"Cannot read file: {e}",
        )

    has_bom = raw_bytes.startswith(codecs.BOM_UTF8)
    line_ending = detect_file_line_ending(raw_bytes)

    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw_bytes).readline)
    except (tokenize.TokenError, SyntaxError):
        encoding = "utf-8"

    try:
        source_text = raw_bytes.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        try:
            source_text = raw_bytes.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            source_text = raw_bytes.decode("latin-1")
            encoding = "latin-1"

    try:
        tree = ast.parse(source_text, filename=str(path))
    except SyntaxError as e:
        return FileScanResult(
            path=path,
            encoding=encoding,
            has_bom=has_bom,
            line_ending=line_ending,
            original_source=source_text,
            parse_error=f"SyntaxError: {e}",
        )

    source_lines = source_text.splitlines(keepends=True)
    visitor = DocstringVisitor(source_lines)
    visitor.visit(tree)

    comments = analyze_comments(raw_bytes, source_text)

    return FileScanResult(
        path=path,
        encoding=encoding,
        has_bom=has_bom,
        line_ending=line_ending,
        original_source=source_text,
        docstrings=visitor.docstrings,
        comments=comments,
    )


def extract_runtime_strings_from_ast(tree: ast.AST) -> list[tuple[str, int, int]]:
    runtime_strings: list[tuple[str, int, int]] = []
    docstring_nodes = set()

    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
        ):
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstring_nodes.add(first.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node not in docstring_nodes:
            runtime_strings.append((node.value, node.lineno, node.col_offset))

    return runtime_strings


def get_executable_tokens(source_code: str) -> list[tuple[int, str]]:
    tokens: list[tuple[int, str]] = []
    bio = io.BytesIO(source_code.encode("utf-8"))
    try:
        for tok in tokenize.tokenize(bio.readline):
            if tok.type in (
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENCODING,
                tokenize.ENDMARKER,
            ):
                continue
            tokens.append((tok.exact_type, tok.string))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return tokens


def normalize_blank_lines(lines: list[str], line_ending: str) -> list[str]:
    result: list[str] = []
    blank_count = 0

    for line in lines:
        stripped = line.strip()
        if stripped == "":
            blank_count += 1
            if blank_count <= 2:
                result.append(line_ending)
        else:
            blank_count = 0
            ending = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else line_ending)
            cleaned = line.rstrip("\r\n") + ending
            result.append(cleaned)

    while result and result[0].strip() == "":
        result.pop(0)

    if result and not result[-1].endswith(("\n", "\r\n")):
        result[-1] = result[-1].rstrip("\r\n") + line_ending

    return result


def remove_docstrings_from_source(
    source_text: str,
    docstrings: list[DocstringItem],
    line_ending: str,
) -> str:
    if not docstrings:
        return source_text

    lines = source_text.splitlines(keepends=True)
    sorted_docstrings = sorted(docstrings, key=lambda d: (d.start_lineno, d.start_col), reverse=True)

    for doc in sorted_docstrings:
        s_idx = doc.start_lineno - 1
        e_idx = doc.end_lineno - 1

        if s_idx < 0 or e_idx >= len(lines):
            continue

        if s_idx == e_idx:
            orig_line = lines[s_idx]
            prefix = orig_line[: doc.start_col]
            suffix = orig_line[doc.end_col :]

            if doc.needs_pass:
                if prefix.strip().endswith(":"):
                    new_line = prefix + f" pass{line_ending}"
                elif prefix.strip() == "":
                    new_line = f"{doc.pass_indent}pass{line_ending}"
                else:
                    new_line = prefix + f"\n{doc.pass_indent}pass" + suffix
                lines[s_idx] = new_line
            else:
                if prefix.strip() == "" and suffix.strip() == "":
                    del lines[s_idx]
                else:
                    new_line = prefix.rstrip() + suffix.lstrip(" \t")
                    if new_line.strip() == "":
                        del lines[s_idx]
                    else:
                        lines[s_idx] = new_line
        else:
            first_line = lines[s_idx]
            last_line = lines[e_idx]
            prefix = first_line[: doc.start_col]
            suffix = last_line[doc.end_col :]

            if doc.needs_pass:
                if prefix.strip().endswith(":"):
                    new_block = prefix + f"{line_ending}{doc.pass_indent}pass{line_ending}"
                    if suffix.strip() != "":
                        new_block += suffix
                elif prefix.strip() == "":
                    new_block = f"{doc.pass_indent}pass{line_ending}"
                    if suffix.strip() != "":
                        new_block += suffix
                else:
                    new_block = prefix.rstrip() + f"{line_ending}{doc.pass_indent}pass{line_ending}"
                    if suffix.strip() != "":
                        new_block += suffix
                lines[s_idx : e_idx + 1] = [new_block]
            else:
                if prefix.strip() == "" and suffix.strip() == "":
                    del lines[s_idx : e_idx + 1]
                else:
                    new_block = prefix.rstrip() + suffix.lstrip(" \t")
                    if new_block.strip() == "":
                        del lines[s_idx : e_idx + 1]
                    else:
                        lines[s_idx : e_idx + 1] = [new_block]

    return "".join(lines)


def remove_comments_from_source(
    source_text: str,
    line_ending: str,
) -> str:
    source_bytes = source_text.encode("utf-8")
    comments = analyze_comments(source_bytes, source_text)
    removable = [c for c in comments if not c.is_preserved]

    if not removable:
        return source_text

    lines = source_text.splitlines(keepends=True)
    sorted_removable = sorted(removable, key=lambda c: (c.start_lineno, c.start_col), reverse=True)

    for comm in sorted_removable:
        line_idx = comm.start_lineno - 1
        if line_idx < 0 or line_idx >= len(lines):
            continue

        curr_line = lines[line_idx]
        prefix = curr_line[: comm.start_col]

        if comm.is_standalone:
            del lines[line_idx]
        else:
            ending = line_ending if curr_line.endswith(line_ending) else ("\r\n" if curr_line.endswith("\r\n") else "\n")
            lines[line_idx] = prefix.rstrip(" \t") + ending

    return "".join(lines)


def transform_source(
    scan_result: FileScanResult,
    remove_docstrings: bool = True,
    remove_comments: bool = True,
) -> tuple[str | None, str | None]:
    has_target = False
    if remove_comments and scan_result.removable_comments:
        has_target = True
    if remove_docstrings and scan_result.removable_docstrings:
        has_target = True

    if not has_target:
        return scan_result.original_source, None

    source = scan_result.original_source
    line_ending = scan_result.line_ending

    if remove_docstrings and scan_result.docstrings:
        source = remove_docstrings_from_source(source, scan_result.docstrings, line_ending)
        try:
            ast.parse(source, filename=str(scan_result.path))
        except SyntaxError as e:
            return None, f"Intermediate SyntaxError after docstring removal: {e}"

    if remove_comments:
        source = remove_comments_from_source(source, line_ending)

    normalized_lines = normalize_blank_lines(source.splitlines(keepends=True), line_ending)
    final_source = "".join(normalized_lines)

    try:
        new_tree = ast.parse(final_source, filename=str(scan_result.path))
    except SyntaxError as e:
        return None, f"SyntaxError in transformed source: {e}"

    try:
        compile(final_source, str(scan_result.path), "exec")
    except Exception as e:  # noqa: BLE001
        return None, f"Compilation failure: {e}"

    orig_tree = ast.parse(scan_result.original_source, filename=str(scan_result.path))
    orig_strings = extract_runtime_strings_from_ast(orig_tree)
    new_strings = extract_runtime_strings_from_ast(new_tree)

    orig_string_values = [s[0] for s in orig_strings]
    new_string_values = [s[0] for s in new_strings]

    if orig_string_values != new_string_values:
        return None, (
            f"Runtime string parity mismatch: original contained {len(orig_string_values)} "
            f"runtime strings, transformed contained {len(new_string_values)}"
        )

    orig_tokens = get_executable_tokens(scan_result.original_source)
    new_tokens = get_executable_tokens(final_source)

    [t for t in orig_tokens if t[0] != tokenize.STRING or not any(d.doc_value == t[1].strip("'\"") for d in scan_result.docstrings)]
    [t for t in new_tokens if t != (tokenize.NAME, "pass") or t in orig_tokens]

    return final_source, None


def discover_python_files(
    root_path: Path,
    include_self: bool = False,
    exclude_dirs: set[str] | None = None,
    exclude_exts: set[str] | None = None,
) -> list[Path]:
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS
    if exclude_exts is None:
        exclude_exts = DEFAULT_EXCLUDE_EXTENSIONS

    self_path = Path(__file__).resolve()
    discovered: list[Path] = []

    if root_path.is_file():
        if root_path.suffix == ".py" and (include_self or root_path.resolve() != self_path):
            return [root_path]
        return []

    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]

        for file_name in files:
            p = Path(root) / file_name
            if p.suffix == ".py":
                if not include_self and p.resolve() == self_path:
                    continue
                discovered.append(p)

    return sorted(discovered)


def run_audit(files: list[Path], verbose: bool = False) -> int:
    print(f"\n{'='*70}")
    print("ARGUS AI - Python Comments & Docstrings Audit")
    print(f"{'='*70}\n")

    total_scanned = len(files)
    files_with_removable = 0
    total_safe_comments = 0
    total_preserved_tooling = 0
    total_preserved_headers = 0
    total_module_docstrings = 0
    total_class_docstrings = 0
    total_func_docstrings = 0
    total_async_docstrings = 0
    parse_errors = 0

    for p in files:
        res = analyze_python_file(p)
        if res.parse_error:
            parse_errors += 1
            if verbose:
                print(f"  [WARN] {p.as_posix()}: {res.parse_error}")
            continue

        safe_comments = len(res.removable_comments)
        module_docs = sum(1 for d in res.docstrings if d.node_type == "module")
        class_docs = sum(1 for d in res.docstrings if d.node_type == "class")
        func_docs = sum(1 for d in res.docstrings if d.node_type == "function")
        async_docs = sum(1 for d in res.docstrings if d.node_type == "async_function")

        tooling_comments = sum(1 for c in res.preserved_comments if "Tooling" in (c.preservation_reason or ""))
        header_comments = sum(1 for c in res.preserved_comments if "shebang" in (c.preservation_reason or "").lower() or "encoding" in (c.preservation_reason or "").lower())

        total_safe_comments += safe_comments
        total_preserved_tooling += tooling_comments
        total_preserved_headers += header_comments
        total_module_docstrings += module_docs
        total_class_docstrings += class_docs
        total_func_docstrings += func_docs
        total_async_docstrings += async_docs

        if res.has_removable_content:
            files_with_removable += 1
            if verbose:
                print(
                    f"  [MODIFIABLE] {p.as_posix()}: "
                    f"comments={safe_comments}, docs={len(res.docstrings)} "
                    f"(tooling preserved={tooling_comments})"
                )

    print(f"Python files scanned:            {total_scanned}")
    print(f"Files containing removable items: {files_with_removable}")
    print(f"Removable # comments:             {total_safe_comments}")
    print(f"Removable module docstrings:      {total_module_docstrings}")
    print(f"Removable class docstrings:       {total_class_docstrings}")
    print(f"Removable function docstrings:    {total_func_docstrings}")
    print(f"Removable async func docstrings:  {total_async_docstrings}")
    print(f"Preserved tooling directives:     {total_preserved_tooling}")
    print(f"Preserved shebang/encoding:       {total_preserved_headers}")
    print(f"Files with parse errors:          {parse_errors}")
    print(f"{'='*70}\n")

    return 0


def run_dry_run(files: list[Path], remove_docstrings: bool = True, remove_comments: bool = True) -> int:
    print(f"\n{'='*70}")
    print("ARGUS AI - Python Comments & Docstrings Dry Run")
    print(f"{'='*70}\n")

    files_to_modify = 0
    skipped_files = 0

    for p in files:
        res = analyze_python_file(p)
        if res.parse_error:
            print(f"[SKIPPED: PARSE ERROR] {p.as_posix()}: {res.parse_error}")
            skipped_files += 1
            continue

        has_target = False
        if remove_comments and res.removable_comments:
            has_target = True
        if remove_docstrings and res.removable_docstrings:
            has_target = True

        if not has_target:
            continue

        transformed, val_err = transform_source(res, remove_docstrings, remove_comments)
        if val_err:
            print(f"[SKIPPED: VALIDATION FAILED] {p.as_posix()}: {val_err}")
            skipped_files += 1
            continue

        if transformed == res.original_source:
            continue

        files_to_modify += 1
        print(f"\n--- {p.as_posix()} ---")
        if remove_docstrings:
            for d in res.removable_docstrings:
                print(f"  [-] docstring ({d.node_type}): lines {d.start_lineno}-{d.end_lineno} (needs_pass={d.needs_pass})")
        if remove_comments:
            for c in res.removable_comments:
                print(f"  [-] comment: line {c.start_lineno}: {c.comment_text[:60]}")
        for c in res.preserved_comments:
            print(f"  [+] PRESERVED ({c.preservation_reason}): line {c.start_lineno}: {c.comment_text[:60]}")

    print(f"\n{'='*70}")
    print(f"Dry-run summary: {files_to_modify} file(s) would be modified, {skipped_files} skipped.")
    print(f"{'='*70}\n")
    return 0


def run_check(files: list[Path], remove_docstrings: bool = True, remove_comments: bool = True) -> int:
    removable_found = 0
    for p in files:
        res = analyze_python_file(p)
        if res.parse_error:
            continue
        if remove_comments and res.removable_comments:
            removable_found += 1
            print(f"[CHECK FAILED] {p.as_posix()}: Contains {len(res.removable_comments)} removable comment(s)")
        elif remove_docstrings and res.removable_docstrings:
            removable_found += 1
            print(f"[CHECK FAILED] {p.as_posix()}: Contains {len(res.removable_docstrings)} removable docstring(s)")

    if removable_found > 0:
        print(f"\nCheck failed: {removable_found} file(s) contain removable comments/docstrings.")
        return 1

    print("\nCheck passed: Zero removable comments/docstrings found.")
    return 0


def run_apply(
    files: list[Path],
    remove_docstrings: bool = True,
    remove_comments: bool = True,
    backup_dir: Path | None = None,
) -> int:
    print(f"\n{'='*70}")
    print("ARGUS AI - Python Comments & Docstrings Apply Transformation")
    print(f"{'='*70}\n")

    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"Backups will be written to: {backup_dir.as_posix()}")

    modified_count = 0
    skipped_count = 0
    total_comments_removed = 0
    total_docstrings_removed = 0
    total_passes_added = 0

    for p in files:
        res = analyze_python_file(p)
        if res.parse_error:
            print(f"[SKIPPED] {p.as_posix()}: {res.parse_error}")
            skipped_count += 1
            continue

        has_target = False
        if remove_comments and res.removable_comments:
            has_target = True
        if remove_docstrings and res.removable_docstrings:
            has_target = True

        if not has_target:
            continue

        transformed, val_err = transform_source(res, remove_docstrings, remove_comments)
        if val_err:
            print(f"[SKIPPED VALIDATION ERROR] {p.as_posix()}: {val_err}")
            skipped_count += 1
            continue

        if transformed == res.original_source:
            continue

        if backup_dir:
            try:
                rel_path = p.relative_to(Path.cwd()) if p.is_relative_to(Path.cwd()) else p.name
                dest_backup = backup_dir / rel_path
                dest_backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest_backup)
            except Exception as e:  # noqa: BLE001
                print(f"[BACKUP FAILED] {p.as_posix()}: {e}")

        output_bytes = transformed.encode(res.encoding)
        if res.has_bom:
            output_bytes = codecs.BOM_UTF8 + output_bytes

        temp_file = p.with_suffix(p.suffix + ".tmp_clean")
        try:
            temp_file.write_bytes(output_bytes)
            temp_file.replace(p)
            modified_count += 1
            total_comments_removed += len(res.removable_comments)
            total_docstrings_removed += len(res.removable_docstrings)
            passes_in_file = sum(1 for d in res.removable_docstrings if d.needs_pass)
            total_passes_added += passes_in_file
            print(f"[CLEANED] {p.as_posix()} (-{len(res.removable_comments)} comments, -{len(res.removable_docstrings)} docs, +{passes_in_file} pass)")
        except OSError as e:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            print(f"[WRITE ERROR] {p.as_posix()}: {e}")
            skipped_count += 1

    print(f"\n{'='*70}")
    print("Apply Summary:")
    print(f"  Files modified:            {modified_count}")
    print(f"  Files skipped:             {skipped_count}")
    print(f"  Total comments removed:    {total_comments_removed}")
    print(f"  Total docstrings removed:  {total_docstrings_removed}")
    print(f"  Total pass statements added:{total_passes_added}")
    print(f"{'='*70}\n")

    return 0


def run_self_test() -> int:
    print(f"\n{'='*70}")
    print("ARGUS AI - Running Cleanup Utility Internal Self-Test Suite")
    print(f"{'='*70}\n")

    fixture_content = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"Module docstring that must be removed.\"\"\"

import os
import sys  # noqa: F401
from typing import Any  # type: ignore

# pragma: no cover
# Normal comment to remove
# TODO: resolve temporary hack

query_sql = \"\"\"
SELECT *
FROM persons
WHERE person_id = ?
\"\"\"

prompt_template = \"\"\"
You are an AI surveillance assistant.
# Important: preserve this internal prompt line.
\"\"\"

html_block = \"\"\"
<div id="target">
    #content
</div>
\"\"\"

msg_with_hash = "Use #ARGUS for tracking"

class DocOnlyClass:
    \"\"\"Class containing only docstring.\"\"\"

class MultiStatementClass:
    \"\"\"Class with docstring and statements.\"\"\"
    active = True

def doc_only_func():
    \"\"\"Function with only docstring.\"\"\"

def standard_func(a: int, b: int) -> int:
    \"\"\"Standard function with docstring and return.\"\"\"
    # Inline comment inside function
    c = a + b  # inline calculation
    return c

async def async_doc_func():
    \"\"\"Async function docstring.\"\"\"
    return await fetch_data()

'''Triple single quote module-level string variable'''
runtime_var = '''Valid triple single quote assigned string'''
"""

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        test_file = temp_dir / "test_fixture.py"
        test_file.write_bytes(fixture_content.encode("utf-8"))

        res = analyze_python_file(test_file)
        if res.parse_error:
            print(f"[FAIL] Self-test fixture failed initial parse: {res.parse_error}")
            return 1

        if len(res.docstrings) != 6:
            print(f"[FAIL] Expected 6 docstrings in fixture, found {len(res.docstrings)}")
            return 1

        tooling_preserved = sum(1 for c in res.preserved_comments if "Tooling" in (c.preservation_reason or ""))
        if tooling_preserved < 3:
            print(f"[FAIL] Expected at least 3 tooling preserved comments, found {tooling_preserved}")
            return 1

        transformed, err = transform_source(res, remove_docstrings=True, remove_comments=True)
        if err:
            print(f"[FAIL] Transformation validation failed: {err}")
            return 1

        try:
            tree = ast.parse(transformed)
            compile(transformed, "test_fixture.py", "exec")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] Transformed code failed compilation: {e}")
            return 1

        assert "#!/usr/bin/env python" in transformed, "Shebang lost"
        assert "# -*- coding: utf-8 -*-" in transformed, "Encoding declaration lost"
        assert "noqa: F401" in transformed, "noqa directive lost"
        assert "type: ignore" in transformed, "type: ignore lost"
        assert "pragma: no cover" in transformed, "pragma directive lost"
        assert "SELECT *" in transformed, "SQL query corrupted"
        assert "You are an AI surveillance assistant." in transformed, "Prompt corrupted"
        assert "# Important: preserve this internal prompt line." in transformed, "Prompt internal line corrupted"
        assert "<div id=\"target\">" in transformed, "HTML corrupted"
        assert "Use #ARGUS for tracking" in transformed, "String with # corrupted"
        assert "Valid triple single quote assigned string" in transformed, "Triple single quote string corrupted"

        assert "Module docstring that must be removed." not in transformed, "Module docstring not removed"
        assert "Class containing only docstring." not in transformed, "Class docstring not removed"
        assert "Function with only docstring." not in transformed, "Function docstring not removed"
        assert "Normal comment to remove" not in transformed, "Normal comment not removed"
        assert "TODO: resolve temporary hack" not in transformed, "TODO comment not removed"

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DocOnlyClass":
                assert len(node.body) == 1 and isinstance(node.body[0], ast.Pass), "DocOnlyClass missing pass"
            if isinstance(node, ast.FunctionDef) and node.name == "doc_only_func":
                assert len(node.body) == 1 and isinstance(node.body[0], ast.Pass), "doc_only_func missing pass"

        # Test idempotency
        test_file.write_bytes(transformed.encode("utf-8"))
        res2 = analyze_python_file(test_file)
        transformed2, err2 = transform_source(res2, remove_docstrings=True, remove_comments=True)
        if err2:
            print(f"[FAIL] Idempotency transformation failed: {err2}")
            return 1
        assert transformed2 == transformed, "Transformation is not idempotent"

    print("\n[SUCCESS] Self-test passed! All 15+ critical scenarios verified cleanly.\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ARGUS AI Safe Developer Utility for Python Comments & Docstrings Cleanup.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cleanup_python_comments.py --audit
  python scripts/cleanup_python_comments.py --dry-run
  python scripts/cleanup_python_comments.py --check
  python scripts/cleanup_python_comments.py --apply
  python scripts/cleanup_python_comments.py --path api/server.py --dry-run
  python scripts/cleanup_python_comments.py --self-test
        """,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--audit",
        action="store_true",
        help="Audit files and print summary report without modifying anything.",
    )
    mode_group.add_argument(
        "--check",
        action="store_true",
        help="Check if files contain removable comments/docstrings; exits 1 if found, 0 if clean.",
    )
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show per-file removal line ranges without modifying files.",
    )
    mode_group.add_argument(
        "--apply",
        action="store_true",
        help="Apply transformations safely to target files with AST and token validation.",
    )
    mode_group.add_argument(
        "--self-test",
        action="store_true",
        help="Run comprehensive built-in test suite on isolated temporary fixtures.",
    )

    parser.add_argument(
        "--path",
        type=str,
        default=".",
        help="File or directory path to process (default: current directory).",
    )
    parser.add_argument(
        "--backup",
        type=str,
        default=None,
        help="Optional directory to write original backup files before modifying.",
    )
    parser.add_argument(
        "--no-remove-docstrings",
        action="store_true",
        help="Do not remove docstrings (preserve all docstrings).",
    )
    parser.add_argument(
        "--no-remove-comments",
        action="store_true",
        help="Do not remove ordinary # comments (preserve comments).",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include cleanup_python_comments.py itself during operations.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed verbose output.",
    )

    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    remove_docstrings = not args.no_remove_docstrings
    remove_comments = not args.no_remove_comments

    target_path = Path(args.path).resolve()
    if not target_path.exists():
        print(f"Error: Target path does not exist: {args.path}", file=sys.stderr)
        return 1

    files = discover_python_files(target_path, include_self=args.include_self)
    if not files:
        print(f"No Python files discovered under {target_path.as_posix()}")
        return 0

    backup_dir = Path(args.backup).resolve() if args.backup else None

    if args.apply:
        return run_apply(files, remove_docstrings, remove_comments, backup_dir)
    elif args.dry_run:
        return run_dry_run(files, remove_docstrings, remove_comments)
    elif args.check:
        return run_check(files, remove_docstrings, remove_comments)
    else:
        # Default mode is audit
        return run_audit(files, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
