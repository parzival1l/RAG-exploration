"""Interactive query interface for the RAG pipeline."""
import sys
import json
from typing import Optional
from src.rag_pipeline import create_rag_pipeline


class QueryInterface:
    """Interactive interface for querying the RAG system."""

    def __init__(self):
        """Initialize the query interface."""
        print("Initializing RAG pipeline...")
        self.pipeline = create_rag_pipeline()
        print("RAG pipeline ready!")

        # Display collection stats
        stats = self.pipeline.get_collection_stats()
        print(f"\nCollection: {stats['collection_name']}")
        print(f"Documents indexed: {stats['points_count']}")
        print(f"Status: {stats['status']}")

    def format_result(self, result: dict, show_sources: bool = True) -> str:
        """Format query result for display.

        Args:
            result: Query result dictionary
            show_sources: Whether to display source documents

        Returns:
            Formatted result string
        """
        output = []
        output.append("=" * 80)
        output.append(f"QUESTION: {result['question']}")
        output.append("=" * 80)
        output.append(f"\nANSWER:\n{result['answer']}\n")
        output.append(f"Sources used: {result['num_sources']}")

        if show_sources and "sources" in result and result["sources"]:
            output.append("\n" + "-" * 80)
            output.append("SOURCE DOCUMENTS:")
            output.append("-" * 80)

            for i, source in enumerate(result["sources"], 1):
                output.append(f"\n[Source {i}]")

                # Display metadata
                metadata = source.get("metadata", {})
                if metadata:
                    output.append("Metadata:")
                    for key, value in metadata.items():
                        if key != "source_id":  # Skip internal IDs
                            output.append(f"  {key}: {value}")

                # Display content (truncated if too long)
                content = source.get("content", "")
                if len(content) > 500:
                    content = content[:500] + "..."
                output.append(f"\nContent:\n{content}")
                output.append("")

        output.append("=" * 80)
        return "\n".join(output)

    def query(self, question: str, show_sources: bool = True) -> dict:
        """Execute a query.

        Args:
            question: User question
            show_sources: Whether to include source documents

        Returns:
            Query result
        """
        result = self.pipeline.query(question, return_sources=show_sources)
        return result

    def interactive_mode(self):
        """Run interactive query mode."""
        print("\n" + "=" * 80)
        print("RAG QUERY INTERFACE - Interactive Mode")
        print("=" * 80)
        print("\nCommands:")
        print("  - Type your question and press Enter")
        print("  - Type 'sources on' to show source documents")
        print("  - Type 'sources off' to hide source documents")
        print("  - Type 'stats' to show collection statistics")
        print("  - Type 'exit' or 'quit' to exit")
        print("=" * 80)

        show_sources = True

        while True:
            try:
                question = input("\n🔍 Your question: ").strip()

                if not question:
                    continue

                # Handle commands
                if question.lower() in ["exit", "quit", "q"]:
                    print("\nGoodbye!")
                    break

                elif question.lower() == "sources on":
                    show_sources = True
                    print("✓ Source display enabled")
                    continue

                elif question.lower() == "sources off":
                    show_sources = False
                    print("✓ Source display disabled")
                    continue

                elif question.lower() == "stats":
                    stats = self.pipeline.get_collection_stats()
                    print(f"\nCollection Statistics:")
                    print(f"  Name: {stats['collection_name']}")
                    print(f"  Points: {stats['points_count']}")
                    print(f"  Vectors: {stats['vectors_count']}")
                    print(f"  Status: {stats['status']}")
                    continue

                # Execute query
                print("\nProcessing query...")
                result = self.query(question, show_sources=show_sources)
                print("\n" + self.format_result(result, show_sources=show_sources))

            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again or type 'exit' to quit.")

    def batch_query(self, questions: list, output_file: Optional[str] = None):
        """Process multiple queries in batch.

        Args:
            questions: List of questions to process
            output_file: Optional file to save results
        """
        results = []

        print(f"\nProcessing {len(questions)} queries...")

        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Processing: {question}")
            try:
                result = self.query(question, show_sources=False)
                results.append(result)
                print(f"✓ Answer: {result['answer'][:100]}...")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                results.append({
                    "question": question,
                    "answer": f"ERROR: {str(e)}",
                    "num_sources": 0
                })

        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {output_file}")

        return results


def main():
    """Main entry point for the query interface."""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Pipeline Query Interface")
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Single question to ask (non-interactive mode)"
    )
    parser.add_argument(
        "--batch", "-b",
        type=str,
        help="File containing questions (one per line) for batch processing"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for batch results (JSON format)"
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't display source documents"
    )

    args = parser.parse_args()

    # Initialize interface
    interface = QueryInterface()

    # Single question mode
    if args.question:
        result = interface.query(args.question, show_sources=not args.no_sources)
        print("\n" + interface.format_result(result, show_sources=not args.no_sources))

    # Batch mode
    elif args.batch:
        with open(args.batch, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]
        interface.batch_query(questions, output_file=args.output)

    # Interactive mode
    else:
        interface.interactive_mode()


if __name__ == "__main__":
    main()
