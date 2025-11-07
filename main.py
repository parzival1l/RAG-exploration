"""Main entry point for the RAG Pipeline application."""
import sys
import argparse
from config.settings import validate_config


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline - Retrieval-Augmented Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index documents into Qdrant
  python main.py index

  # Query the RAG system (interactive mode)
  python main.py query

  # Query with a specific question
  python main.py query -q "What is Qdrant?"

  # Batch query from file
  python main.py query -b questions.txt -o results.json

  # Check collection statistics
  python main.py stats

Note: RAGAS evaluation is banked for later use.
      See src/evaluation.py to re-enable.
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser(
        "index",
        help="Load and index documents into Qdrant"
    )
    index_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the collection (deletes existing data)"
    )

    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Query the RAG system"
    )
    query_parser.add_argument(
        "-q", "--question",
        type=str,
        help="Single question to ask"
    )
    query_parser.add_argument(
        "-b", "--batch",
        type=str,
        help="File with questions (one per line)"
    )
    query_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file for batch results"
    )
    query_parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't display source documents"
    )

    # Evaluate command (RAGAS - Banked for later use)
    # Uncomment when ready to use RAGAS evaluation
    # eval_parser = subparsers.add_parser(
    #     "evaluate",
    #     help="Evaluate RAG pipeline with RAGAS"
    # )
    # eval_parser.add_argument(
    #     "--generate-test-set",
    #     action="store_true",
    #     help="Generate a sample test set"
    # )
    # eval_parser.add_argument(
    #     "--test-set",
    #     type=str,
    #     help="Path to test set JSON file"
    # )
    # eval_parser.add_argument(
    #     "--output",
    #     type=str,
    #     default="evaluation_results.json",
    #     help="Output file for results"
    # )
    # eval_parser.add_argument(
    #     "--num-questions",
    #     type=int,
    #     default=10,
    #     help="Number of questions for test set"
    # )

    # Stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show collection statistics"
    )

    args = parser.parse_args()

    # Show help if no command provided
    if not args.command:
        parser.print_help()
        return

    # Validate configuration
    try:
        validate_config()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)

    # Execute command
    try:
        if args.command == "index":
            from src.data_loader import DataLoader
            print("\n🔄 Starting data indexing pipeline...")
            loader = DataLoader()
            loader.run_full_pipeline(recreate_collection=args.recreate)
            print("\n✓ Indexing completed successfully!")

        elif args.command == "query":
            from src.query_interface import QueryInterface
            interface = QueryInterface()

            if args.question:
                # Single question mode
                result = interface.query(args.question, show_sources=not args.no_sources)
                print("\n" + interface.format_result(result, show_sources=not args.no_sources))

            elif args.batch:
                # Batch mode
                with open(args.batch, 'r') as f:
                    questions = [line.strip() for line in f if line.strip()]
                interface.batch_query(questions, output_file=args.output)

            else:
                # Interactive mode
                interface.interactive_mode()

        # elif args.command == "evaluate":
        #     # RAGAS evaluation - Banked for later use
        #     from src.evaluation import RAGEvaluator
        #     evaluator = RAGEvaluator()
        #
        #     if args.generate_test_set:
        #         test_set_file = "test_set.json"
        #         print(f"\n📝 Generating test set with {args.num_questions} questions...")
        #         evaluator.generate_test_set(
        #             num_questions=args.num_questions,
        #             output_file=test_set_file
        #         )
        #         print(f"\n✓ Test set generated: {test_set_file}")
        #         print("\n💡 Next steps:")
        #         print("  1. Review and customize the test set")
        #         print(f"  2. Run: python main.py evaluate --test-set {test_set_file}")
        #
        #     elif args.test_set:
        #         print(f"\n📊 Loading test set: {args.test_set}")
        #         test_set = evaluator.load_test_set(args.test_set)
        #         results = evaluator.run_evaluation(test_set, output_file=args.output)
        #         evaluator.print_results(results)
        #
        #     else:
        #         print("❌ Please specify --generate-test-set or --test-set <file>")
        #         eval_parser.print_help()

        elif args.command == "stats":
            from src.rag_pipeline import create_rag_pipeline
            print("\n📊 Fetching collection statistics...")
            pipeline = create_rag_pipeline()
            stats = pipeline.get_collection_stats()

            print("\n" + "=" * 60)
            print("QDRANT COLLECTION STATISTICS")
            print("=" * 60)
            print(f"Collection Name: {stats['collection_name']}")
            print(f"Total Points:    {stats['points_count']}")
            print(f"Total Vectors:   {stats['vectors_count']}")
            print(f"Status:          {stats['status']}")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠ Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
