"""Example script demonstrating LinkedIn Post Agent usage."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from linkedin_post_generation import LinkedInPostAgent


def example_basic():
    """Basic example - generate a post about RAG."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Post Generation (RAG Topic)")
    print("="*80 + "\n")

    agent = LinkedInPostAgent()

    result = agent.generate_post(
        topic="What I learned about Retrieval-Augmented Generation",
        style="professional",
        tone="informative",
        length="medium",
        verbose=True,
    )

    return result


def example_code_focused():
    """Example focusing on code repository analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Code-Focused Post Generation")
    print("="*80 + "\n")

    agent = LinkedInPostAgent(repo_root=".")

    result = agent.generate_post(
        topic="Building a modular AI system with Python",
        style="technical",
        tone="informative",
        length="medium",
        force_code_analysis=True,  # Force code analysis
        use_rag=False,  # Disable RAG
        verbose=True,
    )

    return result


def example_storytelling():
    """Example with storytelling style."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Storytelling Post")
    print("="*80 + "\n")

    agent = LinkedInPostAgent()

    result = agent.generate_post(
        topic="My journey learning AI and machine learning",
        style="storytelling",
        tone="inspirational",
        length="long",
        force_rag=True,
        verbose=True,
    )

    return result


def example_custom_settings():
    """Example with custom settings and saving to file."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Settings with File Export")
    print("="*80 + "\n")

    agent = LinkedInPostAgent()

    result = agent.generate_post(
        topic="Why every developer should learn about vector databases",
        style="casual",
        tone="thought-provoking",
        length="short",
        include_hashtags=True,
        force_rag=True,
        verbose=True,
    )

    # Save to file
    output_path = Path("output/linkedin_posts/example_post.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result['post'])

    print(f"\nPost saved to: {output_path}")

    return result


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("LinkedIn Post Agent - Examples")
    print("="*80)

    examples = [
        ("Basic RAG Topic", example_basic),
        ("Code Analysis", example_code_focused),
        ("Storytelling", example_storytelling),
        ("Custom Settings", example_custom_settings),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  {len(examples) + 1}. Run all examples")
    print("  0. Exit")

    choice = input("\nSelect an example (0-{}): ".format(len(examples) + 1)).strip()

    try:
        choice = int(choice)
    except ValueError:
        print("Invalid choice. Exiting.")
        return

    if choice == 0:
        print("Exiting.")
        return
    elif choice == len(examples) + 1:
        # Run all examples
        for name, func in examples:
            try:
                func()
                input("\nPress Enter to continue to next example...")
            except Exception as e:
                print(f"Error in {name}: {e}")
                import traceback
                traceback.print_exc()
    elif 1 <= choice <= len(examples):
        # Run selected example
        name, func = examples[choice - 1]
        try:
            func()
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()

