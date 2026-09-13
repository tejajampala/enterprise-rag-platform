from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN = {"langchain", "langgraph", "langsmith"}


def test_core_and_ui_do_not_import_ragas_transitive_frameworks() -> None:
    paths = [
        *Path("src/consumer_finance_rag").glob("*.py"),
        *Path("app").glob("*.py"),
    ]
    violations: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".", maxsplit=1)[0] in FORBIDDEN:
                    violations.append(f"{path}:{getattr(node, 'lineno', 0)}:{name}")
    assert not violations, "Forbidden architecture imports:\n" + "\n".join(violations)
