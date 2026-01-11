"""Small CLI wrapper to query the codebase agent from command line."""
from __future__ import annotations

import argparse
from .agent import CodebaseAgent


def simple_llm_stub(prompt: str) -> str:
    # Very small heuristic 'LLM' to extract lines that mention the question tokens.
    # For production, replace with a real LLM call.
    qline = "Question:".lower()
    lines = []
    for part in prompt.split("\n\n"):
        if qline in part.lower():
            lines.append(part)
    # Fallback: return prompt tail
    return "\n\n".join(lines) or prompt[-2000:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Question about the repository codebase")
    parser.add_argument("--root", default=".")
    parser.add_argument("--max-files", type=int, default=500)
    args = parser.parse_args()

    agent = CodebaseAgent(root_path=args.root, llm_answer=simple_llm_stub)
    print("Building index (this may take a moment)...")
    agent.build_index(max_files=args.max_files)
    ans = agent.answer_question(args.question)
    print("\nAnswer:\n")
    print(ans)


if __name__ == "__main__":
    main()

