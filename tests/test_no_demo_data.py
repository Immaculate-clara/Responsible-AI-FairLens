"""Quality Gate Test: Enforces CRITICAL RULE — Zero Demo Data & Zero Fake Output.

Fails if:
1. Any .csv, .tsv, .xlsx, .xls data files exist outside the tests/ directory.
2. Any data/ folder exists in the project repository.
3. Any hardcoded demo metrics or fake output triggers exist in pages.
"""

import os
import glob

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_no_data_files_outside_tests():
    """Verify that NO dataset files (.csv, .xlsx, etc.) exist anywhere in the codebase outside tests/."""
    forbidden_extensions = ["*.csv", "*.xlsx", "*.xls", "*.tsv", "*.parquet", "*.feather"]
    data_files = []

    for ext in forbidden_extensions:
        found = glob.glob(os.path.join(REPO_ROOT, "**", ext), recursive=True)
        # Filter out anything in .venv or tests
        filtered = [
            f for f in found
            if ".venv" not in f and "site-packages" not in f and "tests" not in f
        ]
        data_files.extend(filtered)

    assert len(data_files) == 0, (
        f"CRITICAL VIOLATION: Discovered forbidden dataset files outside tests/: {data_files}"
    )


def test_no_data_directory_exists():
    """Verify that no data/ or sample_data/ directory exists in the project root."""
    forbidden_dirs = ["data", "sample_data", "demo_data", "datasets"]
    found_dirs = []

    for d in forbidden_dirs:
        target_path = os.path.join(REPO_ROOT, d)
        if os.path.isdir(target_path):
            found_dirs.append(target_path)

    assert len(found_dirs) == 0, (
        f"CRITICAL VIOLATION: Discovered forbidden sample data folder: {found_dirs}"
    )


def test_no_hardcoded_sample_buttons_in_pages():
    """Verify that no 'Load Sample Data' or 'Try Demo' buttons exist in page source files."""
    pages_dir = os.path.join(REPO_ROOT, "pages")
    assert os.path.isdir(pages_dir)

    forbidden_patterns = [
        "load_sample",
        "try sample data",
        "load demo data",
        "load_demo",
        "sample dataset",
    ]

    violations = []
    for root, _, files in os.walk(pages_dir):
        for f in files:
            if f.endswith(".py"):
                file_path = os.path.join(root, f)
                with open(file_path, "r", encoding="utf-8") as source_file:
                    content = source_file.read().lower()
                    for pattern in forbidden_patterns:
                        if pattern in content:
                            violations.append(f"{f}: found forbidden phrase '{pattern}'")

    assert len(violations) == 0, (
        f"CRITICAL VIOLATION: Pages contain fake data or demo buttons: {violations}"
    )
