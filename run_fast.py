#!/usr/bin/env python3
"""
Run the pipeline with reduced API calls for faster processing.
"""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import BookPipeline


def main():
    """Run pipeline with reduced API calls."""
    print("LongPage Dataset Generator - Fast Mode")
    print("=====================================")
    print("Using reduced API calls for faster processing...")
    print()
    
    # Initialize pipeline
    pipeline = BookPipeline()
    
    # Temporarily reduce embedding dimensions
    from agents.scene_analysis_agent import scene_analysis_agent
    original_dimensions = scene_analysis_agent.embedding_dimensions
    
    # Use only 3 dimensions instead of 7, and reduce calls from 16 to 4
    scene_analysis_agent.embedding_dimensions = ["action", "dialog", "pacing"]
    
    try:
        # Process books
        output_file = pipeline.run_pipeline(max_books=3, output_format="jsonl")
        
        print(f"\nProcessing complete!")
        print(f"Output saved to: {output_file}")
        
        # Show some stats
        with open(output_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f if line.strip())
        
        print(f"Books processed: {line_count}")
        
        # Validate output
        results = pipeline.validate_output(output_file)
        print(f"Valid books: {results['valid_books']}")
        
        if results['issues']:
            print(f"Issues found: {len(results['issues'])}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore original dimensions
        scene_analysis_agent.embedding_dimensions = original_dimensions


if __name__ == "__main__":
    main()