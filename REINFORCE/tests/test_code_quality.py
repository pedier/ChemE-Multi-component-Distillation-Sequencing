"""
Last modified time: 2026-09-23-22:12
Last modified content: Port executable code-style checks into the REINFORCE project
Last modified by: OpenAI Codex
File design: Executable code-style contract
File purpose: Enforce user requirements not covered by mini-linter
File creator: OpenAI Codex
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_METADATA = (
    "Last modified time:",
    "Last modified content:",
    "Last modified by:",
    "File design:",
    "File purpose:",
    "File creator:",
)


def test_python_files_include_metadata_and_respect_file_limit() -> None:
    """Check metadata and the 500-line limit for every Python file.

    Inputs:
        Every Python file under the src and tests directories.
    Returns:
        None. The test passes when metadata and file lengths comply.
    """

    python_files = sorted((PROJECT_ROOT / "src").rglob("*.py"))
    python_files.extend(sorted((PROJECT_ROOT / "tests").rglob("*.py")))
    assert python_files

    # Every file must include all metadata fields and contain no more than 500 lines.
    for path in python_files:
        source = path.read_text(encoding="utf-8")
        for field in REQUIRED_METADATA:
            assert field in source[:500], f"{path} 缺少元信息 {field}"
        assert len(source.splitlines()) <= 500, f"{path} 超过500行"


def test_source_callables_have_docstrings_and_respect_function_limit() -> None:
    """Check docstrings and the 50-line limit for runtime callables.

    Inputs:
        The AST of every Python module under the src directory.
    Returns:
        None. The test passes when every callable meets both requirements.
    """

    source_files = sorted((PROJECT_ROOT / "src").rglob("*.py"))
    assert source_files

    # The AST check covers functions, async functions, and classes with precise locations.
    for path in source_files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert ast.get_docstring(tree), f"{path} 缺少模块 docstring"
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                assert ast.get_docstring(node), f"{path}:{node.lineno} {node.name} 缺少 docstring"
                docstring = ast.get_docstring(node)
                assert "Inputs:" in docstring, f"{path}:{node.lineno} {node.name} 缺少 Inputs"
                assert "Returns:" in docstring, f"{path}:{node.lineno} {node.name} 缺少 Returns"
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno + 1
                assert length <= 50, f"{path}:{node.lineno} {node.name} 超过50行"


def test_python_comments_and_docstrings_are_english() -> None:
    """Check that Python comments and docstrings contain no Chinese characters.

    Inputs:
        Every Python file under the src and tests directories.
    Returns:
        None. The test passes when all comments and docstrings are English.
    """

    python_files = sorted((PROJECT_ROOT / "src").rglob("*.py"))
    python_files.extend(sorted((PROJECT_ROOT / "tests").rglob("*.py")))

    # Tokenization finds comments while AST traversal identifies every module and callable docstring.
    for path in python_files:
        source = path.read_text(encoding="utf-8")
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        comments = [token.string for token in tokens if token.type == tokenize.COMMENT]
        tree = ast.parse(source)
        docstring_nodes = [
            node
            for node in [tree, *ast.walk(tree)]
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        docstrings = [ast.get_docstring(node) for node in docstring_nodes if ast.get_docstring(node)]
        for text in [*comments, *docstrings]:
            has_chinese = any("\u4e00" <= character <= "\u9fff" for character in text)
            assert not has_chinese, f"{path} contains a non-English comment or docstring: {text}"


