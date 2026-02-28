"""Test script for .agentignore functionality"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.code_analysis.indexer import build_index, _read_agentignore, _should_ignore


def test_agentignore():
    """Test .agentignore functionality"""

    # Create a temporary directory structure for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Testing in temporary directory: {tmpdir}")

        # Create test files and directories
        test_files = {
            "main.py": "# Main application file\nprint('Hello')",
            "test_main.py": "# Test file\nimport unittest",
            "data/data.csv": "col1,col2\n1,2",
            "build/output.txt": "Build output",
            "__pycache__/cache.pyc": "cache",
            "venv/lib/python.py": "venv file",
            "docs/README.md": "# Documentation",
            "src/utils.py": "# Utility functions",
            "src/config.pyc": "compiled python",
        }

        # Create files
        for file_path, content in test_files.items():
            full_path = os.path.join(tmpdir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)

        # Create .agentignore file
        agentignore_content = """# Test ignore file
*.pyc
test_*.py
data/
build/
"""
        with open(os.path.join(tmpdir, ".agentignore"), 'w') as f:
            f.write(agentignore_content)

        # Test reading .agentignore
        patterns = _read_agentignore(tmpdir)
        print(f"\nLoaded patterns: {patterns}")
        assert "*.pyc" in patterns
        assert "test_*.py" in patterns
        assert "data/" in patterns
        assert "build/" in patterns

        # Test _should_ignore
        print("\nTesting _should_ignore:")
        test_cases = [
            (os.path.join(tmpdir, "main.py"), False, "main.py should NOT be ignored"),
            (os.path.join(tmpdir, "test_main.py"), True, "test_main.py should be ignored"),
            (os.path.join(tmpdir, "src/config.pyc"), True, "*.pyc should be ignored"),
            (os.path.join(tmpdir, "data/data.csv"), True, "data/ should be ignored"),
            (os.path.join(tmpdir, "build/output.txt"), True, "build/ should be ignored"),
            (os.path.join(tmpdir, "src/utils.py"), False, "src/utils.py should NOT be ignored"),
        ]

        for path, should_ignore, description in test_cases:
            result = _should_ignore(path, tmpdir, patterns)
            status = "✓" if result == should_ignore else "✗"
            print(f"  {status} {description}: {result}")
            assert result == should_ignore, f"Failed: {description}"

        # Test build_index
        print("\nBuilding index...")
        index = build_index(tmpdir)

        print(f"\nIndexed {len(index)} files:")
        for file_path in sorted(index.keys()):
            rel_path = os.path.relpath(file_path, tmpdir)
            print(f"  - {rel_path}")

        # Verify ignored files are not in index
        indexed_relative_paths = [os.path.relpath(p, tmpdir) for p in index.keys()]

        # These should NOT be in the index
        should_not_index = ["test_main.py", "data/data.csv", "build/output.txt", "src/config.pyc"]
        for path in should_not_index:
            assert path not in indexed_relative_paths, f"{path} should not be indexed but was found"
            print(f"  ✓ {path} correctly ignored")

        # These SHOULD be in the index
        should_index = ["main.py", "src/utils.py"]
        for path in should_index:
            assert path in indexed_relative_paths, f"{path} should be indexed but was not found"
            print(f"  ✓ {path} correctly indexed")

        print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_agentignore()

