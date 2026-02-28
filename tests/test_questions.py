"""Test script to ask predefined questions and log answers."""

import os
import sys
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from answer_question import RAGQuestionAnswerer


def main():
    """Run test questions and log results."""

    # Test questions
    questions = [
        "What are the production 'Do's' for RAG?",
        "What is the difference between standard retrieval and the ColPali approach?",
        "Why is hybrid search better than vector-only search?"
    ]

    # Initialize RAG system
    print("=" * 80)
    print("RAG Question Answering Test")
    print("=" * 80)
    print(f"\nStarting test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        rag = RAGQuestionAnswerer()
    except Exception as e:
        error_msg = f"Failed to initialize RAG system: {e}"
        print(error_msg)

        # Log error
        log_path = os.path.join(os.path.dirname(__file__), "test_results.log")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Test run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"ERROR: {error_msg}\n")

        print(f"\nError logged to: {log_path}")
        return

    # Results storage
    results = []

    # Process each question
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 80}")
        print(f"Question {i}/{len(questions)}")
        print(f"{'=' * 80}")

        try:
            result = rag.answer_question(
                question=question,
                show_sources=True
            )
            results.append({
                'question': question,
                'answer': result['answer'],
                'sources': result['sources'],
                'chunks_used': result['chunks_used'],
                'success': True
            })
        except Exception as e:
            print(f"Error answering question: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'question': question,
                'answer': None,
                'error': str(e),
                'success': False
            })

    # Write results to log file
    log_path = os.path.join(os.path.dirname(__file__), "test_results.log")

    print(f"\n{'=' * 80}")
    print(f"Writing results to log file: {log_path}")
    print(f"{'=' * 80}\n")

    with open(log_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("RAG Question Answering Test Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total questions: {len(questions)}\n")
        f.write(f"Successful: {sum(1 for r in results if r['success'])}\n")
        f.write(f"Failed: {sum(1 for r in results if not r['success'])}\n")
        f.write("=" * 80 + "\n\n")

        # Individual results
        for i, result in enumerate(results, 1):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Question {i}\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"Q: {result['question']}\n\n")

            if result['success']:
                f.write(f"A: {result['answer']}\n\n")

                # Write sources
                f.write("Sources:\n")
                f.write("-" * 80 + "\n")

                # Get unique sources
                sources_dict = {}
                for source_info in result['sources']:
                    source = source_info['source']
                    score = source_info['score']
                    if source not in sources_dict or score > sources_dict[source]:
                        sources_dict[source] = score

                for source, score in sorted(sources_dict.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  - {source} (relevance: {score:.2f})\n")

                f.write(f"\nChunks used: {result['chunks_used']}\n")
            else:
                f.write(f"ERROR: {result.get('error', 'Unknown error')}\n")

            f.write("\n")

    print(f"✅ Test completed!")
    print(f"📄 Results saved to: {log_path}")
    print(f"\nSummary:")
    print(f"  Total questions: {len(questions)}")
    print(f"  Successful: {sum(1 for r in results if r['success'])}")
    print(f"  Failed: {sum(1 for r in results if not r['success'])}")


if __name__ == "__main__":
    main()

