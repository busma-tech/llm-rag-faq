"""Zero-dependency AST architecture and public facade tests for ``llm_rag_faq``."""

from __future__ import annotations

import ast
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
PKG_ROOT = SRC_ROOT / "llm_rag_faq"

FORBIDDEN_RAG_SDKS = frozenset(
    {
        "requests",
        "minsearch",
        "fastembed",
        "llama_cpp",
        "anthropic",
        "huggingface_hub",
    }
)

ALLOWED_LAYER_IMPORTS: dict[str, frozenset[str]] = {
    "config": frozenset(),
    "data": frozenset({"config", "data"}),
    "retrieval": frozenset({"config", "retrieval"}),
    "llm": frozenset({"config", "llm"}),
    "rag": frozenset({"config", "llm", "rag"}),
    "cli": frozenset({"config", "data", "retrieval", "llm", "rag"}),
    "api": frozenset({"config", "data", "retrieval", "llm", "rag", "api"}),
    "facade": frozenset({"config", "data", "retrieval", "llm", "rag"}),
}

EXPECTED_PUBLIC_EXPORTS = [
    "AnthropicClient",
    "HybridFAQSearch",
    "LocalGGUFClient",
    "RAGBase",
    "RAGResult",
    "build_index",
    "load_faq_data",
]


@dataclass(frozen=True)
class ImportEdge:
    """Single resolved import edge extracted from a Python module AST."""

    source: str
    target: str
    raw: str
    lineno: int
    file: Path


def discover_modules(
    src_root: Path,
    package_name: str = "llm_rag_faq",
) -> dict[str, Path]:
    """Map canonical dotted module names to file paths under ``src_root/package_name``."""
    pkg_dir = src_root / package_name
    modules: dict[str, Path] = {}
    for py_file in sorted(pkg_dir.rglob("*.py")):
        rel_parts = list(py_file.relative_to(src_root).with_suffix("").parts)
        if py_file.name == "__init__.py":
            mod_name = ".".join(rel_parts[:-1])
        else:
            mod_name = ".".join(rel_parts)
        modules[mod_name] = py_file
    return modules


def is_internal_module(target: str, package_name: str = "llm_rag_faq") -> bool:
    """Return True when ``target`` refers to ``package_name`` or a submodule."""
    return target == package_name or target.startswith(f"{package_name}.")


def _layer_of(mod_name: str, package_name: str = "llm_rag_faq") -> str | None:
    if mod_name == package_name:
        return "facade"
    if mod_name == f"{package_name}.config":
        return "config"
    if mod_name == f"{package_name}.cli":
        return "cli"
    for layer in ("data", "retrieval", "llm", "rag", "api"):
        prefix = f"{package_name}.{layer}"
        if mod_name == prefix or mod_name.startswith(f"{prefix}."):
            return layer
    return None


def _append_from_import_edges(
    edges: list[ImportEdge],
    *,
    mod_name: str,
    base: str,
    raw_prefix: str,
    names: list[ast.alias],
    lineno: int,
    py_file: Path,
    modules: dict[str, Path],
    package_name: str,
) -> None:
    if not names:
        edges.append(
            ImportEdge(
                source=mod_name,
                target=base,
                raw=raw_prefix,
                lineno=lineno,
                file=py_file,
            )
        )
        return
    for alias in names:
        sub = f"{base}.{alias.name}" if base and alias.name != "*" else ""
        target = sub if sub in modules else base
        raw = f"{raw_prefix} import {alias.name}"
        edges.append(
            ImportEdge(
                source=mod_name,
                target=target,
                raw=raw,
                lineno=lineno,
                file=py_file,
            )
        )
        if base == package_name and sub in modules and mod_name != package_name:
            edges.append(
                ImportEdge(
                    source=mod_name,
                    target=base,
                    raw=raw,
                    lineno=lineno,
                    file=py_file,
                )
            )


def extract_import_edges(
    modules: dict[str, Path],
    package_name: str = "llm_rag_faq",
) -> tuple[list[ImportEdge], list[str]]:
    """Parse every module AST and return all import edges plus invalid-relative errors."""
    edges: list[ImportEdge] = []
    errors: list[str] = []

    for mod_name, py_file in modules.items():
        source_text = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(py_file))
        is_init = py_file.name == "__init__.py"
        package_parts = mod_name.split(".") if is_init else mod_name.split(".")[:-1]

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    edges.append(
                        ImportEdge(
                            source=mod_name,
                            target=alias.name,
                            raw=alias.name,
                            lineno=node.lineno,
                            file=py_file,
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    base = node.module or ""
                    _append_from_import_edges(
                        edges,
                        mod_name=mod_name,
                        base=base,
                        raw_prefix=f"from {base}",
                        names=node.names,
                        lineno=node.lineno,
                        py_file=py_file,
                        modules=modules,
                        package_name=package_name,
                    )
                else:
                    up = node.level - 1
                    if up >= len(package_parts):
                        errors.append(
                            f"{py_file}:{node.lineno}: {mod_name} has relative import "
                            f"level={node.level} escaping package {package_name}"
                        )
                        continue
                    anchor = package_parts[: len(package_parts) - up]
                    mod_parts = node.module.split(".") if node.module else []
                    base = ".".join([*anchor, *mod_parts])
                    _append_from_import_edges(
                        edges,
                        mod_name=mod_name,
                        base=base,
                        raw_prefix=f"from {'.' * node.level}{node.module or ''}",
                        names=node.names,
                        lineno=node.lineno,
                        py_file=py_file,
                        modules=modules,
                        package_name=package_name,
                    )
    return edges, errors


def find_circular_imports(
    src_root: Path,
    package_name: str = "llm_rag_faq",
) -> list[list[str]]:
    """Detect internal module import cycles via 3-color DFS."""
    modules = discover_modules(src_root, package_name=package_name)
    edges, _ = extract_import_edges(modules, package_name=package_name)
    graph: dict[str, list[str]] = {mod: [] for mod in modules}
    for edge in edges:
        if edge.target in modules and edge.target not in graph[edge.source]:
            graph[edge.source].append(edge.target)
    for mod in graph:
        graph[mod].sort()

    state: dict[str, int] = {mod: 0 for mod in modules}  # 0=unvisited, 1=visiting, 2=done
    stack: list[str] = []
    cycles: list[list[str]] = []

    def dfs(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for nxt in graph[node]:
            if state[nxt] == 1:
                idx = stack.index(nxt)
                cycles.append([*stack[idx:], nxt])
            elif state[nxt] == 0:
                dfs(nxt)
        stack.pop()
        state[node] = 2

    for mod in sorted(modules):
        if state[mod] == 0:
            dfs(mod)

    return cycles


def find_layer_violations(
    src_root: Path,
    package_name: str = "llm_rag_faq",
) -> list[str]:
    """Return human-readable diagnostics for any layer boundary violations."""
    modules = discover_modules(src_root, package_name=package_name)
    edges, violations = extract_import_edges(modules, package_name=package_name)

    for mod_name, py_file in modules.items():
        if _layer_of(mod_name, package_name=package_name) is None:
            violations.append(f"{py_file}: Unrecognized module layer: {mod_name}")

    for edge in edges:
        if not is_internal_module(edge.target, package_name=package_name):
            continue
        if edge.target not in modules:
            violations.append(
                f"{edge.file}:{edge.lineno}: {edge.source} imports unknown internal "
                f"module {edge.target} ({edge.raw})"
            )
            continue

        src_layer = _layer_of(edge.source, package_name=package_name)
        tgt_layer = _layer_of(edge.target, package_name=package_name)
        if src_layer is None or tgt_layer is None:
            continue

        allowed = ALLOWED_LAYER_IMPORTS[src_layer]
        if tgt_layer not in allowed:
            violations.append(
                f"{edge.file}:{edge.lineno}: {edge.source} ({src_layer}) must not "
                f"import {edge.target} ({tgt_layer})"
            )

    return violations


def find_rag_sdk_leaks(
    src_root: Path,
    package_name: str = "llm_rag_faq",
) -> list[str]:
    """Return diagnostics for any forbidden raw SDK imports inside ``rag/``."""
    modules = discover_modules(src_root, package_name=package_name)
    edges, errors = extract_import_edges(modules, package_name=package_name)
    leaks: list[str] = list(errors)

    for edge in edges:
        if _layer_of(edge.source, package_name=package_name) != "rag":
            continue
        top_pkg = edge.target.split(".", 1)[0]
        if top_pkg in FORBIDDEN_RAG_SDKS:
            leaks.append(
                f"{edge.file}:{edge.lineno}: {edge.source} must not directly import "
                f"infrastructure SDK {top_pkg!r} ({edge.raw})"
            )

    return leaks


def check_no_circular_imports(
    src_root: Path,
    package_name: str = "llm_rag_faq",
) -> None:
    """Assert there are zero circular imports in ``src_root/package_name``."""
    cycles = find_circular_imports(src_root, package_name=package_name)
    if cycles:
        formatted = "; ".join(" -> ".join(cycle) for cycle in cycles)
        raise AssertionError(f"Circular import detected: {formatted}")


def check_layer_rules(
    src_root: Path,
    package_name: str = "llm_rag_faq",
) -> None:
    """Assert strict layer boundaries across all internal modules."""
    violations = find_layer_violations(src_root, package_name=package_name)
    if violations:
        raise AssertionError("Layer boundary violations:\n" + "\n".join(violations))


def check_rag_sdk_isolation(
    src_root: Path,
    package_name: str = "llm_rag_faq",
) -> None:
    """Assert the ``rag`` layer does not directly import raw infrastructure SDKs."""
    leaks = find_rag_sdk_leaks(src_root, package_name=package_name)
    if leaks:
        raise AssertionError("Forbidden SDK imports in rag layer:\n" + "\n".join(leaks))


class TestArchitecture(unittest.TestCase):
    """Verify Option B package layering, acyclicity, SDK isolation, and facade."""

    def test_no_circular_imports(self) -> None:
        modules = discover_modules(SRC_ROOT)
        self.assertGreaterEqual(len(modules), 14)
        check_no_circular_imports(SRC_ROOT)

    def test_layer_dependency_rules(self) -> None:
        check_layer_rules(SRC_ROOT)

    def test_rag_layer_does_not_leak_infrastructure_sdks(self) -> None:
        check_rag_sdk_isolation(SRC_ROOT)

    def test_public_api_exports(self) -> None:
        import llm_rag_faq

        self.assertTrue(hasattr(llm_rag_faq, "__all__"))
        exported = llm_rag_faq.__all__
        self.assertIsInstance(exported, list)
        self.assertEqual(len(exported), len(set(exported)))
        self.assertEqual(exported, sorted(exported))
        self.assertEqual(exported, EXPECTED_PUBLIC_EXPORTS)
        for symbol in exported:
            with self.subTest(symbol=symbol):
                self.assertTrue(hasattr(llm_rag_faq, symbol))
                self.assertIsNotNone(getattr(llm_rag_faq, symbol))

    def test_analyzer_detects_synthetic_violations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src_root = Path(tmp) / "src"
            pkg = src_root / "llm_rag_faq"
            (pkg / "rag").mkdir(parents=True)
            (pkg / "data").mkdir(parents=True)
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "config.py").write_text("import llm_rag_faq.cli\n", encoding="utf-8")
            (pkg / "cli.py").write_text("from llm_rag_faq import RAGBase\n", encoding="utf-8")
            (pkg / "data" / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "data" / "a.py").write_text("from .b import x\n", encoding="utf-8")
            (pkg / "data" / "b.py").write_text("from .a import y\n", encoding="utf-8")
            (pkg / "rag" / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "rag" / "prompts.py").write_text("import requests\n", encoding="utf-8")

            with self.assertRaisesRegex(AssertionError, "Circular import detected"):
                check_no_circular_imports(src_root)
            with self.assertRaisesRegex(AssertionError, "must not import"):
                check_layer_rules(src_root)
            with self.assertRaisesRegex(AssertionError, "requests"):
                check_rag_sdk_isolation(src_root)

        synthetic_layer_cases: list[tuple[str, dict[str, str], str]] = [
            (
                "escaping_relative_config",
                {"config.py": "from .. import cli\n", "cli.py": ""},
                r"llm_rag_faq\.config has relative import level=2 escaping package",
            ),
            (
                "escaping_relative_rag",
                {
                    "rag/__init__.py": "",
                    "rag/prompts.py": "from ... import x\n",
                },
                r"llm_rag_faq\.rag\.prompts has relative import level=3 escaping package",
            ),
            (
                "level2_cross_layer_rag_to_retrieval",
                {
                    "retrieval/__init__.py": "from .hybrid import build_index\n",
                    "retrieval/hybrid.py": "def build_index(): pass\n",
                    "rag/__init__.py": "",
                    "rag/prompts.py": "from ..retrieval import build_index\n",
                },
                r"llm_rag_faq\.rag\.prompts \(rag\) must not import llm_rag_faq\.retrieval",
            ),
            (
                "cli_imports_submod_via_facade",
                {
                    "config.py": "",
                    "cli.py": "from llm_rag_faq import config\n",
                },
                r"llm_rag_faq\.cli \(cli\) must not import llm_rag_faq \(facade\)",
            ),
            (
                "rag_imports_submod_via_facade",
                {
                    "config.py": "",
                    "rag/__init__.py": "",
                    "rag/prompts.py": "from llm_rag_faq import config\n",
                },
                r"llm_rag_faq\.rag\.prompts \(rag\) must not import llm_rag_faq \(facade\)",
            ),
        ]
        for label, files, pattern in synthetic_layer_cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                src_root = Path(tmp) / "src"
                pkg = src_root / "llm_rag_faq"
                pkg.mkdir(parents=True)
                (pkg / "__init__.py").write_text("", encoding="utf-8")
                for rel_path, content in files.items():
                    file_path = pkg / rel_path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(AssertionError, pattern):
                    check_layer_rules(src_root)
