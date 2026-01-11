"""CLI wrapper for LinkedIn Post Generation Agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from linkedin_post_generation.agent import LinkedInPostAgent
import config


def main():
    """Main function to run LinkedIn post generation from CLI."""
    parser = argparse.ArgumentParser(
        description="Generate LinkedIn posts using AI with code analysis and RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a post about a topic (interactive mode)
  python -m agents.linkedin_post.cli
  
  # Generate a post about RAG (will use RAG agent automatically)
  python -m agents.linkedin_post.cli -t "What I learned about RAG systems"
  
  # Generate a post about your code repository
  python -m agents.linkedin_post.cli -t "My journey building an AI project" --force-code
  
  # Generate with specific style and tone
  python -m agents.linkedin_post.cli -t "AI in production" -s technical -T thought-provoking
  
  # Generate a long post with custom settings
  python -m agents.linkedin_post.cli -t "RAG best practices" -l long --force-rag
  
  # Generate without hashtags
  python -m agents.linkedin_post.cli -t "My AI journey" --no-hashtags
        """
    )

    parser.add_argument(
        '-t', '--topic',
        type=str,
        help='Topic for the LinkedIn post (if not provided, will prompt for input)'
    )
    parser.add_argument(
        '-s', '--style',
        type=str,
        default='professional',
        choices=['professional', 'casual', 'technical', 'storytelling'],
        help='Writing style (default: professional)'
    )
    parser.add_argument(
        '-T', '--tone',
        type=str,
        default='informative',
        choices=['informative', 'inspirational', 'thought-provoking'],
        help='Tone of the post (default: informative)'
    )
    parser.add_argument(
        '-l', '--length',
        type=str,
        default='medium',
        choices=['short', 'medium', 'long'],
        help='Post length (default: medium)'
    )
    parser.add_argument(
        '--no-hashtags',
        action='store_true',
        help='Do not include hashtags in the post'
    )
    parser.add_argument(
        '--no-code',
        action='store_true',
        help='Disable code analysis agent'
    )
    parser.add_argument(
        '--no-rag',
        action='store_true',
        help='Disable RAG agent'
    )
    parser.add_argument(
        '--force-code',
        action='store_true',
        help='Force use of code analysis agent regardless of topic'
    )
    parser.add_argument(
        '--force-rag',
        action='store_true',
        help='Force use of RAG agent regardless of topic'
    )
    parser.add_argument(
        '--repo-root',
        type=str,
        default='.',
        help='Root directory of the repository for code analysis (default: current directory)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode - minimal output'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Save the generated post to a file'
    )

    args = parser.parse_args()

    # Get topic from user if not provided
    topic = args.topic
    if not topic:
        print("LinkedIn Post Generator")
        print("=" * 80)
        topic = input("\nEnter the topic for your LinkedIn post: ").strip()
        if not topic:
            print("Error: Topic cannot be empty")
            sys.exit(1)

    # Initialize the agent
    try:
        if not args.quiet:
            print("\nInitializing LinkedIn Post Agent...")

        agent = LinkedInPostAgent(
            repo_root=args.repo_root,
            generator_backend=config.GENERATOR_BACKEND,
        )
    except Exception as e:
        print(f"Error initializing agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Generate the post
    try:
        result = agent.generate_post(
            topic=topic,
            style=args.style,
            tone=args.tone,
            length=args.length,
            include_hashtags=not args.no_hashtags,
            use_code_analysis=not args.no_code,
            use_rag=not args.no_rag,
            force_code_analysis=args.force_code,
            force_rag=args.force_rag,
            verbose=not args.quiet,
        )

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['post'])
            print(f"\nPost saved to: {output_path}")

        # Print summary in quiet mode
        if args.quiet:
            print(result['post'])
        else:
            print("\nPost generation complete!")
            print(f"Subagents used: Code Analysis={result['subagents_used']['code_analysis']}, "
                  f"RAG={result['subagents_used']['rag']}")

        return 0

    except KeyboardInterrupt:
        print("\n\nPost generation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\nError generating post: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

