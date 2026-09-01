import argparse
import ast
import os
import re
import sys
import tempfile
import time
from pathlib import Path

TARGET_FOLDERS = [
    "api",
    "configs",
    "core",
    "enrollment",
    "evaluation",
    "events",
    "intelligence",
    "models",
    "monitoring",
    "pipeline",
    "preprocessing",
    "scripts",
    "security_layer",
    "services",
    "storage",
    "streaming",
    "tests",
    "training",
    "utils",
]

REQUIRED_SECTIONS = [
    "Responsibilities",
    "Key Modules",
    "Data Flow",
    "Configuration",
    "Public Interfaces",
    "Tests",
    "Related Documentation",
]

REQUIRED_SECTIONS_BY_FOLDER = {
    "scripts": [
        "Folder Purpose",
        "Script Inventory",
        "Script Metadata",
        "CLI Reference",
        "Common Commands",
        "Command Index",
        "Script Dependency Graph",
        "Script Execution Order",
        "Generated Outputs",
        "Safety Classification",
        "Script Execution Flow",
        "Dependencies",
        "Cross References",
        "Safety Notes",
        "Command Examples",
        "Automatic Maintenance",
    ],
}


def get_active_files_for_folder(folder_path: Path) -> list[str]:
    folder_name = folder_path.name

    if folder_name == "configs":
        files = [f.name for f in folder_path.iterdir() if f.is_file() and f.suffix in (".yaml", ".yml", ".json")]
    elif folder_name == "scripts":
        files = [
            f.name
            for f in folder_path.iterdir()
            if f.is_file()
            and f.suffix in (".py", ".ps1", ".bat", ".sh")
            and not f.name.startswith("__")
            and f.name != "README.md"
        ]
    elif folder_name == "models":
        items = ["architectures/bygait_light.py"]
        for sub in [
            "active",
            "appearance_gallery",
            "candidates",
            "gallery",
            "live_gallery",
            "reid",
            "rollback",
            "weights",
        ]:
            if (folder_path / sub).exists():
                items.append(f"{sub}/")
        return sorted(items)
    else:
        items = []
        for f in folder_path.iterdir():
            if f.is_file() and f.suffix == ".py" and not f.name.startswith("__") and f.name != "README.md":
                items.append(f.name)
            elif f.is_dir() and not f.name.startswith("__") and not f.name.startswith("."):
                if folder_name == "api" and f.name == "routes":
                    for sub in f.iterdir():
                        if sub.is_file() and sub.suffix == ".py" and not sub.name.startswith("__"):
                            items.append(f"routes/{sub.name}")
                elif folder_name in ("pipeline", "tests"):
                    items.append(f"{f.name}/")

        files = items

    return sorted(files)


def extract_script_description(path: Path) -> str:
    if not path.exists():
        return f"Script {path.name}"

    content = path.read_text(encoding="utf-8", errors="ignore")

    if path.suffix == ".ps1":
        m_syn = re.search(
            r"\.SYNOPSIS\s*\n\s*(.*?)(?=\n\s*\.|\n\s*#>)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if m_syn:
            first_line = m_syn.group(1).strip().splitlines()[0].strip()
            if first_line:
                return first_line

    if path.suffix == ".py":
        try:
            tree = ast.parse(content)
            doc = ast.get_docstring(tree)
            if doc and doc.strip():
                lines = [line.strip() for line in doc.strip().splitlines() if line.strip()]
                if lines and not lines[0].startswith("Usage:"):
                    return lines[0]
        except (SyntaxError, ValueError):
            pass

    if path.suffix == ".py":
        m = re.search(r'ArgumentParser\s*\(\s*description=["\'](.*?)["\']', content, re.DOTALL)
        if m:
            return m.group(1).strip()

    for line in content.splitlines()[:20]:
        l_s = line.strip()
        if l_s.startswith(("#", "REM", "::")):
            clean = re.sub(r"^(#|REM|::|<#|#>\s*)+", "", l_s).strip()
            if (
                clean
                and not clean.startswith("!")
                and not clean.startswith("ARGUS AI Automatic")
                and not clean.startswith("@echo")
            ):
                return clean

    stem = path.stem
    if stem.startswith("test_"):
        component = stem[5:].replace("_", " ")
        return f"Validation test script for {component}."
    elif stem.startswith("evaluate_"):
        task = stem[9:].replace("_", " ")
        return f"Evaluation script for {task}."
    elif stem.startswith("benchmark_") or stem == "benchmark":
        task = stem.replace("_", " ")
        return f"Performance benchmark script for {task}."
    elif stem == "start_system":
        return "System startup launcher script."
    elif stem == "system_check":
        return "Environment and dependency verification script."

    clean_stem = stem.replace("_", " ")
    return f"Utility script for {clean_stem}."


def get_script_primary_usage(path: Path) -> str:
    name = path.name
    if path.suffix == ".ps1":
        return f"powershell -ExecutionPolicy Bypass -File scripts/{name}"
    elif path.suffix in (".bat", ".sh"):
        return f"scripts/{name}"
    elif name.startswith("test_"):
        return f"pytest scripts/{name}"
    else:
        return f"python scripts/{name}"


def _ast_node_to_value(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.List):
        return [_ast_node_to_value(el) for el in node.elts]
    if isinstance(node, ast.Tuple):
        return [_ast_node_to_value(el) for el in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _ast_node_to_value(node.operand)
        if isinstance(val, (int, float)):
            return -val
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _ast_node_to_value(node.left)
        right = _ast_node_to_value(node.right)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
    return None


def extract_cli_arguments(path: Path) -> dict | None:
    if path.suffix != ".py" or not path.exists():
        return None

    content = path.read_text(encoding="utf-8", errors="ignore")
    if "ArgumentParser" not in content:
        return None

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    description = ""
    arguments: list[dict] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func

        is_ap = (isinstance(func, ast.Attribute) and func.attr == "ArgumentParser") or (
            isinstance(func, ast.Name) and func.id == "ArgumentParser"
        )
        if is_ap:
            for kw in node.keywords:
                if kw.arg == "description":
                    val = _ast_node_to_value(kw.value)
                    if isinstance(val, str):
                        description = val.strip()
            continue

        if isinstance(func, ast.Attribute) and func.attr == "add_argument":
            arg_info: dict = {
                "flags": [],
                "type": None,
                "default": None,
                "help": None,
                "required": False,
                "choices": None,
                "action": None,
            }

            for pos_arg in node.args:
                val = _ast_node_to_value(pos_arg)
                if isinstance(val, str):
                    arg_info["flags"].append(val)

            for kw in node.keywords:
                if kw.arg == "type":
                    val = _ast_node_to_value(kw.value)
                    if val is not None:
                        arg_info["type"] = str(val)
                elif kw.arg == "default":
                    arg_info["default"] = _ast_node_to_value(kw.value)
                elif kw.arg == "help":
                    val = _ast_node_to_value(kw.value)
                    if isinstance(val, str):
                        arg_info["help"] = val
                elif kw.arg == "choices":
                    val = _ast_node_to_value(kw.value)
                    if isinstance(val, list):
                        arg_info["choices"] = val
                elif kw.arg == "required":
                    val = _ast_node_to_value(kw.value)
                    if isinstance(val, bool):
                        arg_info["required"] = val
                elif kw.arg == "action":
                    val = _ast_node_to_value(kw.value)
                    if isinstance(val, str):
                        arg_info["action"] = val

            if arg_info["flags"]:
                arguments.append(arg_info)

    if not description and not arguments:
        return None

    return {"description": description, "arguments": arguments}


def _get_script_category(name: str) -> str:
    if name == "sync_folder_readmes.py":
        return "Documentation"
    if name == "install_git_hooks.py":
        return "Git"
    if name in (
        "activate_venv.ps1",
        "manage_venv.ps1",
        "bootstrap_env.ps1",
        "download_package.py",
        "process_runner.py",
    ):
        return "Environment"
    if name in ("start_system.bat", "start_system.sh"):
        return "Deployment"
    if name in (
        "preprocess_casia.py",
        "build_gallery.py",
        "clean_live_gallery.py",
        "remove_gallery_identity.py",
        "remove_numeric_gallery_identities.py",
        "set_gallery_identity_status.py",
        "run_auto_enrollment.py",
        "extract_casia_skeletons.py",
    ):
        return "Dataset"
    if name in (
        "export_bygait_onnx.py",
        "export_silhouette_unet_onnx.py",
        "build_tensorrt_engine.py",
        "migrate_output_layout.py",
    ):
        return "Conversion"
    if (
        name.startswith(
            (
                "test_",
                "demo_",
                "run_",
                "generate_",
                "validate_",
                "evaluate_",
                "verify_",
                "simulate_",
                "audit_",
                "benchmark",
            )
        )
        or name
        in (
            "system_check.py",
            "detect_environment.py",
            "verify_environment.py",
            "doctor.py",
            "smoke_test_deployment.py",
        )
    ):
        return "Validation"
    return "Development"


def _get_safety_classification(name: str, path: Path) -> str:
    if name == "sync_folder_readmes.py":
        return "Documentation"
    if name == "install_git_hooks.py":
        return "Git"
    if name in (
        "activate_venv.ps1",
        "manage_venv.ps1",
        "bootstrap_env.ps1",
        "download_package.py",
        "process_runner.py",
    ):
        return "Environment"
    if name in ("start_system.bat", "start_system.sh"):
        return "Deployment"
    if name.startswith("test_") or name == "verify_environment.py":
        return "Validation"
    if name in ("system_check.py", "detect_environment.py"):
        return "Read-Only"

    if not path.exists():
        return "Unknown"

    content = path.read_text(encoding="utf-8", errors="ignore")

    write_indicators = [
        "write_text(",
        "os.replace(",
        "store.save(",
        ".save(",
        "shutil.copy",
        "shutil.move",
        "json.dump(",
        ".savefig(",
    ]
    has_writes = any(indicator in content for indicator in write_indicators)
    has_mkdir = "mkdir(" in content

    if has_writes or has_mkdir:
        return "Repository Modification"

    return "Read-Only"


def _is_auto_start(name: str) -> bool:
    return name in ("activate_venv.ps1", "sync_folder_readmes.py")


def _is_used_by_ci(name: str, root_dir: Path) -> bool:
    workflows_dir = root_dir / ".github" / "workflows"
    if not workflows_dir.exists():
        return False
    for wf_path in workflows_dir.iterdir():
        if wf_path.is_file() and wf_path.suffix in (".yml", ".yaml"):
            try:
                wf_content = wf_path.read_text(encoding="utf-8", errors="ignore")
                if name in wf_content:
                    return True
            except OSError:
                continue
    return False


def _is_used_by_hook(name: str, root_dir: Path) -> bool:
    hook_installer = root_dir / "scripts" / "install_git_hooks.py"
    if not hook_installer.exists():
        return False
    try:
        content = hook_installer.read_text(encoding="utf-8", errors="ignore")
        hook_match = re.search(r'HOOK_CONTENT\s*=\s*"""(.*?)"""', content, re.DOTALL)
        if hook_match:
            return name in hook_match.group(1)
        return name in content
    except OSError:
        return False


def _detect_generated_outputs(name: str, path: Path) -> list[str]:
    if name == "sync_folder_readmes.py":
        return ["*/README.md", "docs/README_INDEX.md"]
    if name == "install_git_hooks.py":
        return [".git/hooks/pre-commit"]
    if name == "activate_venv.ps1":
        return ["No file modifications"]
    if name in ("start_system.bat", "start_system.sh"):
        return ["No file modifications"]
    if name == "system_check.py":
        return ["No file modifications"]
    if name.startswith("test_"):
        return ["No file modifications"]

    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8", errors="ignore")
    outputs: set = set()

    cli_info = extract_cli_arguments(path)
    if cli_info:
        for arg in cli_info.get("arguments", []):
            flags_str = " ".join(arg.get("flags", [])).lower()
            default = arg.get("default")
            if (
                isinstance(default, str)
                and "/" in default
                and any(kw in flags_str for kw in ("output", "engine", "export"))
            ):
                outputs.add(default)

    for match in re.finditer(r"(\w+)\s*=\s*Path\(\s*\"([^\"]+)\"\s*\)", content):
        var_name = match.group(1)
        path_val = match.group(2).replace("\\", "/")
        if f"{var_name}.mkdir" in content:
            outputs.add(path_val)

    if "store.save(" in content:
        for match in re.finditer(r"gallery_dir\s*=\s*\"([^\"]+)\"", content):
            outputs.add(match.group(1))
        for match in re.finditer(r"VectorStore\(\s*gallery_dir\s*=\s*\"([^\"]+)\"", content):
            outputs.add(match.group(1))

    for match in re.finditer(r"(?:OUTPUT|GALLERY)_\w*\s*=\s*\"([^\"]+)\"", content):
        val = match.group(1).replace("\\", "/")
        if "/" in val:
            outputs.add(val)

    if not outputs:
        write_indicators = [
            "write_text(",
            "os.replace(",
            "store.save(",
            ".save(",
            "shutil.copy",
            "shutil.move",
            "json.dump(",
        ]
        has_writes = any(ind in content for ind in write_indicators)
        has_mkdir = "mkdir(" in content
        if not has_writes and not has_mkdir:
            return ["No file modifications"]
        return ["Runtime-determined paths"]

    return sorted(outputs)


def _build_script_dependencies(folder_path: Path, root_dir: Path) -> list[tuple[str, str, str]]:
    active_files = get_active_files_for_folder(folder_path)
    edges: list[tuple[str, str, str]] = []

    for script_name in active_files:
        if script_name == "sync_folder_readmes.py":
            continue
        script_path = folder_path / script_name
        if not script_path.exists():
            continue
        try:
            content = script_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for other_name in active_files:
            if other_name == script_name:
                continue
            if re.search(
                r"(?:^|[\s\"'/,;:()\[\]])" + re.escape(other_name) + r"(?:$|[\s\"'/,;:()\[\]])",
                content,
            ):
                edges.append((script_name, other_name, "reference"))

    workflows_dir = root_dir / ".github" / "workflows"
    if workflows_dir.exists():
        for wf_path in sorted(workflows_dir.iterdir()):
            if wf_path.is_file() and wf_path.suffix in (".yml", ".yaml"):
                try:
                    wf_content = wf_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for script_name in active_files:
                    if script_name in wf_content:
                        edges.append((f"CI: {wf_path.name}", script_name, "ci"))

    if "sync_folder_readmes.py" in active_files:
        edges.append(("sync_folder_readmes.py", "Package READMEs", "output"))
        edges.append(("sync_folder_readmes.py", "docs/README_INDEX.md", "output"))
    if "install_git_hooks.py" in active_files:
        edges.append(("install_git_hooks.py", ".git/hooks/pre-commit", "output"))

    return sorted(edges)


def _generate_metadata_table(folder_path: Path, root_dir: Path) -> str:
    active_files = get_active_files_for_folder(folder_path)
    lines = [
        "| Script | Category | CLI | Auto | Used by CI | Used by Hook | Description |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in active_files:
        script_path = folder_path / name
        category = _get_script_category(name)
        has_cli = "Yes" if extract_cli_arguments(script_path) else "No"
        auto = "Yes" if _is_auto_start(name) else "No"
        ci = "Yes" if _is_used_by_ci(name, root_dir) else "No"
        hook = "Yes" if _is_used_by_hook(name, root_dir) else "No"
        desc = extract_script_description(script_path)
        if len(desc) > 60:
            desc = desc[:57] + "..."
        lines.append(f"| [{name}]({name}) | {category} | {has_cli} | {auto} | {ci} | {hook} | {desc} |")
    return "\n".join(lines)


def _generate_cli_reference(folder_path: Path) -> str:
    active_files = get_active_files_for_folder(folder_path)
    sections: list[str] = []

    for name in active_files:
        script_path = folder_path / name
        cli_info = extract_cli_arguments(script_path)
        if cli_info is None:
            continue

        desc = cli_info.get("description", "") or extract_script_description(script_path)
        args = cli_info.get("arguments", [])
        usage = get_script_primary_usage(script_path)

        block: list[str] = [
            "<details>",
            f"<summary><strong>{name}</strong> — {desc}</summary>",
            "",
            f"**Usage**: `{usage}`",
            "",
        ]

        if args:
            block.extend(
                [
                    "| Flag / Argument | Type | Required | Default | Description |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for arg in args:
                flags = ", ".join(f"`{f}`" for f in arg["flags"])
                arg_type = str(arg.get("type") or "—")
                if arg.get("action") in ("store_true", "store_false"):
                    arg_type = "flag"
                elif arg.get("action") == "append":
                    arg_type = f"{arg_type} (repeatable)"
                required = "Yes" if arg.get("required") else "No"
                default = arg.get("default")
                if default is None:
                    default_str = "None"
                elif isinstance(default, str):
                    default_str = f"`{default}`"
                else:
                    default_str = str(default)
                help_text = arg.get("help") or "—"
                if arg.get("choices"):
                    choices_str = ", ".join(str(c) for c in arg["choices"])
                    help_text += f" (choices: {choices_str})"
                block.append(f"| {flags} | {arg_type} | {required} | {default_str} | {help_text} |")

        block.append("")
        block.append("**Examples**:")
        block.append("")
        block.append("```bash")
        block.append(usage)

        example_parts = [f"python scripts/{name}"]
        for arg in args[:2]:
            if arg["flags"] and arg["flags"][0].startswith("--"):
                flag = arg["flags"][0]
                if arg.get("action") in ("store_true", "store_false"):
                    example_parts.append(flag)
                elif arg.get("default") is not None and arg["default"] != "":
                    example_parts.append(f"{flag} {arg['default']}")
        if len(example_parts) > 1:
            block.append(" ".join(str(p) for p in example_parts))

        block.append("```")
        block.append("")
        block.append("</details>")
        block.append("")

        sections.append("\n".join(block))

    if not sections:
        return "No CLI-enabled scripts detected."

    return "\n".join(sections)


def _generate_command_index(folder_path: Path) -> str:
    active_files = get_active_files_for_folder(folder_path)
    lines = ["| Command | Description |", "| --- | --- |"]
    for name in active_files:
        script_path = folder_path / name
        usage = get_script_primary_usage(script_path)
        desc = extract_script_description(script_path)
        if len(desc) > 70:
            desc = desc[:67] + "..."
        lines.append(f"| `{usage}` | {desc} |")
    return "\n".join(lines)


def _generate_dependency_graph(folder_path: Path, root_dir: Path) -> str:
    edges = _build_script_dependencies(folder_path, root_dir)

    if not edges:
        return "No verified inter-script dependencies detected."

    def node_id(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", name)

    lines = ["```mermaid", "flowchart TD"]

    seen_nodes: set = set()
    for src, tgt, etype in edges:
        src_id = node_id(src)
        tgt_id = node_id(tgt)
        if src_id not in seen_nodes:
            lines.append(f'    {src_id}["{src}"]')
            seen_nodes.add(src_id)
        if tgt_id not in seen_nodes:
            lines.append(f'    {tgt_id}["{tgt}"]')
            seen_nodes.add(tgt_id)
        lines.append(f"    {src_id} -->|{etype}| {tgt_id}")

    lines.append("```")
    return "\n".join(lines)


def _generate_execution_order(folder_path: Path) -> str:
    active_files = get_active_files_for_folder(folder_path)

    category_order = [
        "Environment",
        "Validation",
        "Documentation",
        "Git",
        "Dataset",
        "Conversion",
        "Development",
        "Deployment",
    ]

    by_category: dict[str, list[str]] = {}
    for name in active_files:
        cat = _get_script_category(name)
        by_category.setdefault(cat, []).append(name)

    lines = ["```mermaid", "flowchart TD"]
    prev_id = None
    step = 0
    for cat in category_order:
        if cat not in by_category:
            continue
        step += 1
        cat_id = f"step{step}"
        scripts = sorted(by_category[cat])
        count = len(scripts)
        suffix = f" ({count} scripts)" if count > 1 else ""
        lines.append(f'    {cat_id}["{step}. {cat}{suffix}"]')
        if prev_id:
            lines.append(f"    {prev_id} --> {cat_id}")
        prev_id = cat_id

    lines.append("```")
    return "\n".join(lines)


def _generate_change_impact(folder_path: Path) -> str:
    active_files = get_active_files_for_folder(folder_path)
    lines = ["| Script | Generated / Modified Outputs |", "| --- | --- |"]
    for name in active_files:
        script_path = folder_path / name
        outputs = _detect_generated_outputs(name, script_path)
        output_str = ", ".join(f"`{o}`" for o in outputs)
        lines.append(f"| [{name}]({name}) | {output_str} |")
    return "\n".join(lines)


def _generate_safety_classification_section(folder_path: Path) -> str:
    active_files = get_active_files_for_folder(folder_path)

    classifications: dict[str, list[str]] = {}
    for name in active_files:
        script_path = folder_path / name
        safety = _get_safety_classification(name, script_path)
        classifications.setdefault(safety, []).append(name)

    lines = ["| Classification | Scripts |", "| --- | --- |"]
    for classification in sorted(classifications.keys()):
        scripts = sorted(classifications[classification])
        script_links = ", ".join(f"[{s}]({s})" for s in scripts)
        lines.append(f"| **{classification}** | {script_links} |")

    return "\n".join(lines)


def _generate_cross_references(root_dir: Path) -> str:
    links: list[str] = []

    root_readme = root_dir / "README.md"
    if root_readme.exists():
        links.append("- [Root README](../README.md)")

    index_path = root_dir / "docs" / "README_INDEX.md"
    if index_path.exists():
        links.append("- [Documentation Index](../docs/README_INDEX.md)")

    workflows_dir = root_dir / ".github" / "workflows"
    if workflows_dir.exists():
        for wf_path in sorted(workflows_dir.iterdir()):
            if wf_path.is_file() and wf_path.suffix in (".yml", ".yaml"):
                links.append(f"- [CI: {wf_path.name}](../.github/workflows/{wf_path.name})")

    for folder in TARGET_FOLDERS:
        if folder == "scripts":
            continue
        readme = root_dir / folder / "README.md"
        if readme.exists():
            try:
                content = readme.read_text(encoding="utf-8", errors="ignore")
                if "scripts/" in content:
                    links.append(f"- [{folder}/README.md](../{folder}/README.md)")
            except OSError:
                continue

    return "\n".join(links) if links else "No cross-references detected."


def _sync_script_categories(folder_path: Path, content: str) -> str:
    active_files = get_active_files_for_folder(folder_path)

    categories = {
        "VALIDATION_SCRIPTS": [f for f in active_files if _get_script_category(f) == "Validation"],
        "DATASET_SCRIPTS": [f for f in active_files if _get_script_category(f) == "Dataset"],
        "CONVERSION_SCRIPTS": [f for f in active_files if _get_script_category(f) == "Conversion"],
        "DEVELOPMENT_SCRIPTS": [f for f in active_files if _get_script_category(f) == "Development"],
    }

    for cat_name, cat_files in categories.items():
        m_start = f"<!-- BEGIN SYNC: {cat_name} -->"
        m_end = f"<!-- END SYNC: {cat_name} -->"
        if m_start in content and m_end in content:
            lines = []
            for f in sorted(cat_files):
                file_p = folder_path / f
                desc = extract_script_description(file_p)
                usage = get_script_primary_usage(file_p)
                lines.append(f"- **[{f}]({f})**: {desc} (`{usage}`)")
            cat_str = "\n".join(lines)
            pattern = re.escape(m_start) + r".*?" + re.escape(m_end)
            content = re.sub(pattern, f"{m_start}\n{cat_str}\n{m_end}", content, flags=re.DOTALL)

    return content


def _sync_scripts_extended_sections(folder_path: Path, content: str, root_dir: Path) -> str:
    sync_generators = {
        "SCRIPT_METADATA_TABLE": lambda: _generate_metadata_table(folder_path, root_dir),
        "CLI_REFERENCE": lambda: _generate_cli_reference(folder_path),
        "COMMAND_INDEX": lambda: _generate_command_index(folder_path),
        "SCRIPT_DEPENDENCY_GRAPH": lambda: _generate_dependency_graph(folder_path, root_dir),
        "SCRIPT_EXECUTION_ORDER": lambda: _generate_execution_order(folder_path),
        "CHANGE_IMPACT": lambda: _generate_change_impact(folder_path),
        "SAFETY_CLASSIFICATION": lambda: _generate_safety_classification_section(folder_path),
        "CROSS_REFERENCES": lambda: _generate_cross_references(root_dir),
    }

    for marker_name, generator in sync_generators.items():
        m_start = f"<!-- BEGIN SYNC: {marker_name} -->"
        m_end = f"<!-- END SYNC: {marker_name} -->"
        if m_start in content and m_end in content:
            generated = generator()
            pattern = re.escape(m_start) + r".*?" + re.escape(m_end)
            content = re.sub(
                pattern,
                f"{m_start}\n{generated}\n{m_end}",
                content,
                flags=re.DOTALL,
            )

    return content


def check_folder_readme(folder_path: Path) -> tuple[bool, list[str]]:
    readme_path = folder_path / "README.md"
    issues = []

    if not readme_path.exists():
        return False, [f"Missing {readme_path}"]

    content = readme_path.read_text(encoding="utf-8")

    required_sections = REQUIRED_SECTIONS_BY_FOLDER.get(folder_path.name, REQUIRED_SECTIONS)
    for sec in required_sections:
        if f"## {sec}" not in content:
            issues.append(f"Missing section '## {sec}' in {readme_path.name}")

    active_files = get_active_files_for_folder(folder_path)
    for file_name in active_files:
        base_name = file_name.rstrip("/")
        if base_name not in content:
            issues.append(f"Module/File '{file_name}' not listed in {readme_path.name}")

    return len(issues) == 0, issues


def check_readme_index(root_dir: Path) -> tuple[bool, list[str]]:
    index_path = root_dir / "docs" / "README_INDEX.md"
    issues = []

    if not index_path.exists():
        return False, [f"Missing {index_path}"]

    content = index_path.read_text(encoding="utf-8")
    for folder in TARGET_FOLDERS:
        if f"{folder}/README.md" not in content:
            issues.append(f"Missing entry for {folder}/README.md in docs/README_INDEX.md")

    return len(issues) == 0, issues



def _atomic_write_file(target_path: Path, new_content: str, max_retries: int = 5) -> None:
    target_path = Path(target_path).resolve()


    newline = "\n"
    if target_path.exists():
        try:
            raw_bytes = target_path.read_bytes()
            if b"\r\n" in raw_bytes:
                newline = "\r\n"
        except OSError:
            pass

    fd, tmp_path_str = tempfile.mkstemp(
        dir=str(target_path.parent),
        prefix=".readme_sync_",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_path_str)

    try:
        with open(fd, "w", encoding="utf-8", newline=newline) as tmp_f:
            tmp_f.write(new_content)
            tmp_f.flush()
            try:
                os.fsync(tmp_f.fileno())
            except OSError:
                pass


        for attempt in range(max_retries):
            try:
                os.replace(str(tmp_path), str(target_path))
                return
            except (PermissionError, OSError) as exc:
                if attempt < max_retries - 1:
                    time.sleep(0.05 * (2**attempt))
                else:
                    raise OSError(
                        f"Failed to atomically update {target_path} after {max_retries} attempts: {exc}"
                    ) from exc
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def update_folder_readme(folder_path: Path, root_dir: Path | None = None) -> bool:
    readme_path = folder_path / "README.md"
    if not readme_path.exists():
        return False

    content = readme_path.read_text(encoding="utf-8")
    marker_start = "<!-- BEGIN SYNC: KEY_MODULES -->"
    marker_end = "<!-- END SYNC: KEY_MODULES -->"

    if marker_start not in content or marker_end not in content:
        return False

    active_files = get_active_files_for_folder(folder_path)

    desc_map: dict[str, str] = {}
    usage_map: dict[str, str] = {}
    table_lines = content.split(marker_start)[1].split(marker_end)[0].strip().split("\n")
    for line in table_lines:
        if (
            line.startswith("|")
            and not line.startswith("| Module")
            and not line.startswith("| Script")
            and not line.startswith("|---")
            and not line.startswith("| ---")
            and not line.startswith("|:---")
            and not line.startswith("| :---")
        ):
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) >= 2:
                mod_link = cols[0]
                desc = cols[1]
                m = re.search(r"\[(.*?)\]", mod_link)
                key = m.group(1) if m else mod_link.strip("`")
                desc_map[key] = desc
                if len(cols) >= 3:
                    usage_map[key] = cols[2]

    if folder_path.name == "scripts":
        new_lines = ["| Script | Purpose | Primary Usage |", "| --- | --- | --- |"]
        for f in active_files:
            file_p = folder_path / f
            desc = extract_script_description(file_p)
            usage = usage_map.get(f, f"`{get_script_primary_usage(file_p)}`")
            if not usage.startswith("`"):
                usage = f"`{usage}`"
            link = f"[{f}]({f})"
            new_lines.append(f"| {link} | {desc} | {usage} |")
    else:
        new_lines = ["| Module | Purpose |", "| --- | --- |"]
        for f in active_files:
            desc = desc_map.get(f, f"Module/resource file {f}")
            if "/" in f and not f.endswith("/"):
                link = f"[{f}]({f})"
            elif f.endswith("/"):
                link = f"`{f}`"
            else:
                link = f"[{f}]({f})"
            new_lines.append(f"| {link} | {desc} |")

    new_table_str = "\n".join(new_lines)
    pattern = re.escape(marker_start) + r".*?" + re.escape(marker_end)
    new_content = re.sub(pattern, f"{marker_start}\n{new_table_str}\n{marker_end}", content, flags=re.DOTALL)

    if folder_path.name == "scripts":
        new_content = _sync_script_categories(folder_path, new_content)
        effective_root = root_dir or folder_path.parent
        new_content = _sync_scripts_extended_sections(folder_path, new_content, effective_root)

    norm_content = content.replace("\r\n", "\n")
    norm_new_content = new_content.replace("\r\n", "\n")

    if norm_new_content != norm_content:
        _atomic_write_file(readme_path, norm_new_content)
        return True

    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize ARGUS AI folder README files.")
    parser.add_argument("--check", action="store_true", help="Check if folder READMEs are synchronized (CI mode).")
    parser.add_argument("--update", action="store_true", help="Update folder README key modules tables.")
    parser.add_argument("--root-dir", default=".", help="Root workspace directory.")
    args = parser.parse_args()

    should_update = args.update or not args.check
    root_dir = Path(args.root_dir)
    total_issues = 0
    updated_count = 0

    print("Checking ARGUS AI folder documentation alignment...")

    for folder_name in TARGET_FOLDERS:
        folder_path = root_dir / folder_name
        if not folder_path.exists():
            continue

        if should_update and update_folder_readme(folder_path, root_dir=root_dir):
            print(f"[UPDATED] Synchronized {folder_name}/README.md")
            updated_count += 1

        is_valid, issues = check_folder_readme(folder_path)
        if not is_valid:
            total_issues += len(issues)
            for issue in issues:
                print(f"[WARN] {issue}")
        else:
            print(f"[OK] {folder_name}/README.md is synchronized and valid.")

    idx_valid, idx_issues = check_readme_index(root_dir)
    if not idx_valid:
        total_issues += len(idx_issues)
        for issue in idx_issues:
            print(f"[WARN] {issue}")
    else:
        print("[OK] docs/README_INDEX.md is valid.")

    if args.check and total_issues > 0:
        print(f"\n[FAIL] Synchronization check failed with {total_issues} issue(s).")
        return 1

    print(f"\n[SUCCESS] All package READMEs synchronized cleanly ({updated_count} updated).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
