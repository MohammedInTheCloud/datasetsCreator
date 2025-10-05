#!/usr/bin/env python3
"""
Main entry point for the LongPage Dataset Generator.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import run_pipeline, BookPipeline
from pipeline_parallel import run_pipeline as run_pipeline_parallel, ParallelBookPipeline


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate LongPage-compatible datasets from book texts"
    )
    
    parser.add_argument(
        "--max-books", "-m",
        type=int,
        default=None,
        help="Maximum number of books to process (default: all)"
    )
    
    parser.add_argument(
        "--output-format", "-f",
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format (default: jsonl)"
    )
    
    parser.add_argument(
        "--books-dir", "-b",
        type=str,
        default="D:\\local\\datasetsCreator\\books",
        help="Directory containing book files"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="D:\\local\\datasetsCreator\\output",
        help="Output directory for generated datasets"
    )
    
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate the output file after generation"
    )

    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Use parallel processing for improved performance (default: false)"
    )

    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (use original sequential pipeline)"
    )
    
    args = parser.parse_args()

    # Determine whether to use parallel processing
    use_parallel = args.parallel and not args.no_parallel

    try:
        if use_parallel:
            print("🚀 Using parallel processing pipeline...")
            # Run parallel pipeline
            pipeline = ParallelBookPipeline(
                books_dir=args.books_dir,
                output_dir=args.output_dir,
                enable_parallel=True
            )

            output_file = pipeline.run_pipeline(
                max_books=args.max_books,
                output_format=args.output_format
            )
        else:
            print("📚 Using sequential processing pipeline...")
            # Run original pipeline
            pipeline = BookPipeline(
                books_dir=args.books_dir,
                output_dir=args.output_dir
            )

            output_file = pipeline.run_pipeline(
                max_books=args.max_books,
                output_format=args.output_format
            )
        
        print(f"\n[SUCCESS] Pipeline completed successfully!")
        print(f"Output saved to: {output_file}")
        
        # Validate if requested
        if args.validate:
            print("\nValidating output...")
            results = pipeline.validate_output(output_file)
            
            print(f"\nValidation Results:")
            print(f"   Total books: {results['total_books']}")
            print(f"   Valid books: {results['valid_books']}")
            print(f"   Invalid books: {results['invalid_books']}")
            
            if results['issues']:
                print(f"\nIssues found:")
                for issue in results['issues'][:10]:  # Show first 10 issues
                    print(f"   - {issue}")
                if len(results['issues']) > 10:
                    print(f"   ... and {len(results['issues']) - 10} more")
            else:
                print("   [OK] No issues found!")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()