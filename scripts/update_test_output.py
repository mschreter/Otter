#!/usr/bin/env python3
"""
Update deal.II/CTest reference output files from a verbose ctest log.

Usage
-----

1) Run ctest with verbose output and save the log:

    ctest -V | tee ctest.log

2) Update the reference output files:

    python3 update_test_output.py ctest.log

Optional arguments:

    --dry-run   Print copy operations without modifying files
    --backup    Create *.bak backups before overwriting files
"""

import argparse
import re
import shutil
from pathlib import Path


SOURCE_RE = re.compile(r"DIFF failed\. ------ Source:\s+(.+)")
RESULT_RE = re.compile(r"DIFF failed\. ------ Result:\s+(.+)")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Update deal.II/CTest reference output files from a verbose "
            "ctest log.\n\n"
            "Example usage:\n"
            "  ctest -V | tee ctest.log\n"
            "  python3 update_test_output.py ctest.log"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "ctest_log",
        type=Path,
        help="Path to verbose ctest log file",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print copy operations without modifying files",
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create *.bak backups before overwriting files",
    )

    args = parser.parse_args()

    text = args.ctest_log.read_text(errors="replace")

    # Ensure this is a verbose ctest log
    if "Test command:" not in text:
        raise SystemExit(
            "ctest log does not appear to be generated with verbose output.\n"
            "Please run:\n\n"
            "  ctest -V | tee ctest.log\n"
        )

    lines = text.splitlines()

    current_source = None
    copied = 0

    for line in lines:
        source_match = SOURCE_RE.search(line)
        if source_match:
            current_source = Path(source_match.group(1).strip())
            continue

        result_match = RESULT_RE.search(line)
        if result_match and current_source is not None:
            generated_output = Path(result_match.group(1).strip())
            reference_output = current_source

            if not generated_output.exists():
                print(f"[skip] missing generated output: {generated_output}")
                continue

            if args.backup and reference_output.exists():
                backup = reference_output.parent / (
                    reference_output.name + ".bak"
                )

                print(f"[backup] {reference_output} -> {backup}")

                if not args.dry_run:
                    shutil.copy2(reference_output, backup)

            print(f"[copy] {generated_output}")
            print(f"    -> {reference_output}")

            if not args.dry_run:
                shutil.copy2(generated_output, reference_output)

            copied += 1
            current_source = None

    print(f"\nDone. Updated {copied} output file(s).")


if __name__ == "__main__":
    main()
