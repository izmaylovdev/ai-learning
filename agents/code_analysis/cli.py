"""CLI wrapper to query the code analysis agent from command line."""
from __future__ import annotations

import argparse
from .agent import analyze_code
from .tools import initialize_code_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze code repository using AI agent"
    )
    parser.add_argument("question", help="Question about the repository codebase")
    parser.add_argument("--root", default=".", help="Repository root path (default: current directory)")
    args = parser.parse_args()

    # Initialize code analysis tools with the repository root
    initialize_code_analysis(args.root)

    # Use the analyze_code tool
    print("Analyzing repository...")
    answer = analyze_code.invoke({"question": args.question})

    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()

