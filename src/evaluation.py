"""Evaluation module using RAGAS for the RAG pipeline."""
import json
from typing import List, Dict, Any, Optional
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.rag_pipeline import create_rag_pipeline


class RAGEvaluator:
    """Evaluator for RAG pipeline using RAGAS metrics."""

    def __init__(self):
        """Initialize the evaluator."""
        self.pipeline = create_rag_pipeline()
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    def generate_test_set(
        self,
        num_questions: int = 10,
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate a test set of questions for evaluation.

        This is a template method - you should customize the questions
        based on your specific knowledge base.

        Args:
            num_questions: Number of questions to generate
            output_file: Optional file to save the test set

        Returns:
            List of test cases with questions and ground truth
        """
        # Example test set - customize based on your domain
        test_set = [
            {
                "question": "What is Qdrant?",
                "ground_truth": "Qdrant is a vector similarity search engine and vector database.",
            },
            {
                "question": "How does vector search work?",
                "ground_truth": "Vector search works by converting data into high-dimensional vectors and finding similar vectors using distance metrics.",
            },
            {
                "question": "What are the benefits of using Qdrant?",
                "ground_truth": "Qdrant offers fast similarity search, filtering capabilities, and easy integration with machine learning pipelines.",
            },
        ]

        # Limit to requested number
        test_set = test_set[:num_questions]

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(test_set, f, indent=2)
            print(f"Test set saved to: {output_file}")

        return test_set

    def load_test_set(self, file_path: str) -> List[Dict[str, Any]]:
        """Load test set from a JSON file.

        Args:
            file_path: Path to the test set file

        Returns:
            List of test cases
        """
        with open(file_path, 'r') as f:
            test_set = json.load(f)
        return test_set

    def run_evaluation(
        self,
        test_set: List[Dict[str, Any]],
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run RAGAS evaluation on the test set.

        Args:
            test_set: List of test cases with 'question' and 'ground_truth'
            output_file: Optional file to save results

        Returns:
            Evaluation results
        """
        print(f"Running evaluation on {len(test_set)} test cases...")

        # Prepare data for RAGAS
        questions = []
        ground_truths = []
        answers = []
        contexts = []

        for i, test_case in enumerate(test_set, 1):
            print(f"\n[{i}/{len(test_set)}] Processing: {test_case['question']}")

            # Get RAG response
            result = self.pipeline.query(
                test_case['question'],
                return_sources=True
            )

            questions.append(test_case['question'])
            ground_truths.append(test_case['ground_truth'])
            answers.append(result['answer'])

            # Extract context from sources
            context_list = [
                source['content']
                for source in result.get('sources', [])
            ]
            contexts.append(context_list)

            print(f"  ✓ Answer generated ({len(result.get('sources', []))} sources)")

        # Create RAGAS dataset
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        print("\n" + "=" * 80)
        print("Running RAGAS evaluation...")
        print("=" * 80)

        try:
            # Run evaluation
            results = evaluate(
                dataset=eval_dataset,
                metrics=self.metrics,
            )

            # Format results
            eval_results = {
                "overall_scores": {
                    metric: float(results[metric])
                    for metric in results
                    if not metric.startswith('_')
                },
                "num_test_cases": len(test_set),
                "individual_results": []
            }

            # Add individual results
            for i in range(len(questions)):
                eval_results["individual_results"].append({
                    "question": questions[i],
                    "answer": answers[i],
                    "ground_truth": ground_truths[i],
                    "num_contexts": len(contexts[i]),
                })

            # Save results
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                print(f"\n✓ Results saved to: {output_file}")

            return eval_results

        except Exception as e:
            print(f"\n❌ Error during evaluation: {str(e)}")
            print("\nNote: RAGAS evaluation requires OpenAI API access.")
            print("If you don't have OpenAI API key, you can still use the RAG pipeline,")
            print("but RAGAS evaluation will not be available.")
            raise

    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a formatted way.

        Args:
            results: Evaluation results dictionary
        """
        print("\n" + "=" * 80)
        print("RAGAS EVALUATION RESULTS")
        print("=" * 80)

        print(f"\nTest Cases: {results['num_test_cases']}")
        print("\nOverall Scores:")
        print("-" * 80)

        for metric, score in results['overall_scores'].items():
            # Format metric name
            metric_name = metric.replace('_', ' ').title()
            status = "✓" if score > 0.7 else "⚠" if score > 0.5 else "✗"
            print(f"{status} {metric_name:.<50} {score:.4f}")

        print("\nMetric Targets:")
        print("  Faithfulness: > 0.80 (responses grounded in context)")
        print("  Answer Relevancy: > 0.85 (directly addresses question)")
        print("  Context Precision: > 0.70 (relevant chunks retrieved)")
        print("  Context Recall: > 0.70 (important info not missed)")

        print("\n" + "=" * 80)


def main():
    """Main entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluation")
    parser.add_argument(
        "--generate-test-set",
        action="store_true",
        help="Generate a sample test set"
    )
    parser.add_argument(
        "--test-set",
        type=str,
        help="Path to test set JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=10,
        help="Number of questions for test set generation"
    )

    args = parser.parse_args()

    evaluator = RAGEvaluator()

    # Generate test set
    if args.generate_test_set:
        test_set_file = "test_set.json"
        print(f"Generating test set with {args.num_questions} questions...")
        evaluator.generate_test_set(
            num_questions=args.num_questions,
            output_file=test_set_file
        )
        print(f"\n✓ Test set generated: {test_set_file}")
        print("\nPlease review and customize the test set, then run:")
        print(f"  python src/evaluation.py --test-set {test_set_file}")
        return

    # Load and run evaluation
    if args.test_set:
        test_set = evaluator.load_test_set(args.test_set)
        results = evaluator.run_evaluation(test_set, output_file=args.output)
        evaluator.print_results(results)
    else:
        print("Please specify --generate-test-set or --test-set <file>")
        print("Run with --help for more information")


if __name__ == "__main__":
    main()
