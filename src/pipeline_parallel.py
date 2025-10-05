"""
Parallel-enhanced pipeline orchestrator for the book dataset generator.
Provides dramatic performance improvements through concurrent processing.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from utils.preprocessor import BookPreprocessor
from agents.llm_agent_async import hybrid_llm_agent
from agents.scene_agent import scene_agent
from agents.scene_analysis_agent_parallel import parallel_scene_analysis_agent
from agents.chapter_agent import chapter_agent
from agents.book_agent import book_agent
from agents.synthetic_prompt_generator import synthetic_prompt_generator
from utils.composer import compose
from utils.parallel_processor import parallel_processor, run_async

# Configure logging
log_file = Path(__file__).parent.parent / 'pipeline_parallel.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ],
    force=True  # Override any existing configuration
)
logger = logging.getLogger(__name__)


class ParallelBookPipeline:
    """Enhanced pipeline with parallel processing capabilities."""

    def __init__(self, books_dir: str = None, output_dir: str = None, enable_parallel: bool = True):
        """Initialize the parallel pipeline.

        Args:
            books_dir: Directory containing input books.
            output_dir: Directory for output files.
            enable_parallel: Whether to enable parallel processing.
        """
        self.books_dir = Path(books_dir or "D:\\local\\datasetsCreator\\books")
        self.output_dir = Path(output_dir or "D:\\local\\datasetsCreator\\output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Parallel processing configuration
        self.enable_parallel = enable_parallel and os.getenv("DISABLE_PARALLEL_PROCESSING", "false").lower() != "true"

        # Initialize components
        self.preprocessor = BookPreprocessor(str(self.books_dir))

        # Concurrency limits from environment
        self.max_concurrent_books = int(os.getenv("MAX_CONCURRENT_BOOKS", "3"))
        self.max_concurrent_chapters = int(os.getenv("MAX_CONCURRENT_CHAPTERS", "5"))
        self.max_concurrent_scenes = int(os.getenv("MAX_CONCURRENT_SCENES", "10"))

        logger.info(f"Initialized parallel pipeline")
        logger.info(f"Books directory: {self.books_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Parallel processing: {self.enable_parallel}")
        logger.info(f"Concurrency limits: books={self.max_concurrent_books}, chapters={self.max_concurrent_chapters}, scenes={self.max_concurrent_scenes}")

    async def process_book_async(self, book_path: str) -> Dict[str, Any]:
        """Async version of book processing with parallel stages.

        Args:
            book_path: Path to the book file.

        Returns:
            Complete book data in LongPage format.
        """
        book_name = Path(book_path).name
        logger.info(f"Processing book asynchronously: {book_name}")

        try:
            # Step 1: Load and preprocess (always sequential)
            logger.info("Step 1: Preprocessing book")
            book_data = self.preprocessor.preprocess_book(book_path)

            # Initialize final JSON structure
            final_json = {
                "use_prompt": "",
                "book_highlight": "",
                "book_title": book_data["title"],
                "book_tags": [],
                "book_archetype": "",
                "world_rules": [],
                "story_arcs": [],
                "character_archetypes": {},
                "writing_style": [],
                "book_characters": {
                    "main_characters": [],
                    "side_characters": []
                },
                "book_chapters": {}
            }

            # Step 2: Process chapters in parallel
            logger.info("Step 2: Processing chapters with parallelization")
            processed_chapters = await self._process_chapters_parallel(book_data["chapters"])

            # Add processed chapters to final JSON
            final_json["book_chapters"] = processed_chapters

            # Step 3: Generate book-level metadata (can run concurrently with some steps)
            logger.info("Step 3: Generating book-level metadata")
            book_metadata_input = {
                "chapters": book_data["chapters"],
                "processed_chapters": processed_chapters
            }

            # Extract book metadata
            book_metadata = book_agent.generate_book_metadata(book_metadata_input)
            final_json.update(book_metadata)

            # Step 4: Generate synthetic elements
            logger.info("Step 4: Generating synthetic elements")

            # Generate book highlight
            final_json["book_highlight"] = synthetic_prompt_generator.generate_book_highlight(final_json)

            # Generate archetype if not present
            if not final_json["book_archetype"]:
                final_json["book_archetype"] = self._generate_book_archetype(final_json)

            # Generate tags
            final_json["book_tags"] = self._generate_book_tags(final_json)

            # Generate synthetic prompt
            final_json["use_prompt"] = synthetic_prompt_generator.generate_prompt(final_json)

            logger.info(f"Successfully processed book: {book_name}")
            return final_json

        except Exception as e:
            logger.error(f"Failed to process book {book_name}: {e}", exc_info=True)
            raise

    async def _process_chapters_parallel(self, chapters: Dict[str, str]) -> Dict[str, Dict]:
        """Process all chapters in parallel with controlled concurrency.

        Args:
            chapters: Dictionary mapping chapter titles to chapter text.

        Returns:
            Dictionary of processed chapters with metadata.
        """
        if not self.enable_parallel or len(chapters) == 1:
            # Sequential fallback
            return await self._process_chapters_sequential(chapters)

        logger.info(f"Processing {len(chapters)} chapters in parallel (max {self.max_concurrent_chapters} concurrent)")

        # Create semaphore to limit chapter concurrency
        chapter_semaphore = asyncio.Semaphore(self.max_concurrent_chapters)

        async def process_single_chapter(chapter_title: str, chapter_text: str) -> tuple[str, Dict]:
            """Process a single chapter with rate limiting."""
            async with chapter_semaphore:
                try:
                    logger.debug(f"Processing chapter: {chapter_title}")
                    return await self._process_single_chapter_async(chapter_title, chapter_text)
                except Exception as e:
                    logger.error(f"Failed to process chapter {chapter_title}: {e}")
                    # Return minimal fallback data
                    return chapter_title, {
                        "chapter": chapter_text,
                        "embedding_space": {dim: 50 for dim in ["action", "dialog", "world_building", "exposition", "romantic", "erotic", "pacing"]},
                        "chapter_summary": [f"Processing failed for {chapter_title}"],
                        "scene_breakdown": []
                    }

        # Create tasks for all chapters
        chapter_tasks = [
            process_single_chapter(title, text)
            for title, text in chapters.items()
        ]

        # Process chapters concurrently
        results = await asyncio.gather(*chapter_tasks, return_exceptions=True)

        # Convert results back to dictionary
        processed_chapters = {}
        failed_count = 0

        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                chapter_title, chapter_data = result
                processed_chapters[chapter_title] = chapter_data
            elif isinstance(result, Exception):
                logger.error(f"Chapter processing exception: {result}")
                failed_count += 1

        logger.info(f"Chapter processing complete: {len(processed_chapters)} successful, {failed_count} failed")
        return processed_chapters

    async def _process_single_chapter_async(self, chapter_title: str, chapter_text: str) -> tuple[str, Dict]:
        """Process a single chapter asynchronously.

        Args:
            chapter_title: Title of the chapter.
            chapter_text: Text of the chapter.

        Returns:
            Tuple of (chapter_title, processed_chapter_data).
        """
        # Segment into scenes (sequential for now, could be parallelized too)
        scenes = scene_agent.segment_scenes(chapter_title, chapter_text)

        # Analyze scenes in parallel
        if self.enable_parallel and len(scenes) > 1:
            # Prepare scene data for batch processing
            scenes_data = [
                {
                    "text": scene["text"],
                    "scene_name": scene["scene_name"],
                    "narrative_focus": scene.get("narrative_focus", ""),
                    "narrative_perspective": scene.get("narrative_perspective", "")
                }
                for scene in scenes
            ]

            # Batch analyze scenes
            analyzed_scenes = await parallel_scene_analysis_agent.batch_analyze_scenes(scenes_data)
        else:
            # Sequential scene analysis
            analyzed_scenes = []
            for scene in scenes:
                analyzed_scene = parallel_scene_analysis_agent.analyze_scene(
                    scene["text"],
                    scene["scene_name"],
                    scene.get("narrative_focus", ""),
                    scene.get("narrative_perspective", "")
                )
                analyzed_scenes.append(analyzed_scene)

        # Generate chapter summary
        scene_summaries = [s["scene_summary_short"][0] for s in analyzed_scenes if s["scene_summary_short"]]
        chapter_summary = chapter_agent.generate_chapter_summary(chapter_title, scene_summaries)

        # Aggregate scene embeddings for chapter
        chapter_embedding = parallel_scene_analysis_agent.aggregate_scene_embeddings(analyzed_scenes)

        # Store processed chapter
        processed_chapter = {
            "chapter": chapter_text,
            "embedding_space": chapter_embedding,
            "chapter_summary": chapter_summary,
            "scene_breakdown": analyzed_scenes
        }

        return chapter_title, processed_chapter

    async def _process_chapters_sequential(self, chapters: Dict[str, str]) -> Dict[str, Dict]:
        """Sequential fallback for chapter processing.

        Args:
            chapters: Dictionary mapping chapter titles to chapter text.

        Returns:
            Dictionary of processed chapters with metadata.
        """
        processed_chapters = {}

        for chapter_title, chapter_text in chapters.items():
            logger.debug(f"Processing chapter sequentially: {chapter_title}")
            _, chapter_data = await self._process_single_chapter_async(chapter_title, chapter_text)
            processed_chapters[chapter_title] = chapter_data

        return processed_chapters

    def process_book(self, book_path: str) -> Dict[str, Any]:
        """Synchronous wrapper for book processing.

        Args:
            book_path: Path to the book file.

        Returns:
            Complete book data in LongPage format.
        """
        if self.enable_parallel:
            return run_async(self.process_book_async(book_path))
        else:
            # Use original sequential processing
            return self._process_book_sequential(book_path)

    def _process_book_sequential(self, book_path: str) -> Dict[str, Any]:
        """Original sequential book processing for fallback.

        Args:
            book_path: Path to the book file.

        Returns:
            Complete book data in LongPage format.
        """
        # Import the original pipeline for fallback
        from pipeline import BookPipeline
        original_pipeline = BookPipeline(str(self.books_dir), str(self.output_dir))
        return original_pipeline.process_book(book_path)

    def _generate_book_archetype(self, book_data: Dict) -> str:
        """Generate book archetype using LLM with robust validation.

        Args:
            book_data: Book data dictionary.

        Returns:
            Generated archetype string (guaranteed to be clean).
        """
        # Import the original method from pipeline
        from pipeline import BookPipeline
        original_pipeline = BookPipeline(str(self.books_dir), str(self.output_dir))
        return original_pipeline._generate_book_archetype(book_data)

    def _generate_book_tags(self, book_data: Dict) -> List[str]:
        """Generate book tags based on content.

        Args:
            book_data: Book data dictionary.

        Returns:
            List of tag strings.
        """
        # Import the original method from pipeline
        from pipeline import BookPipeline
        original_pipeline = BookPipeline(str(self.books_dir), str(self.output_dir))
        return original_pipeline._generate_book_tags(book_data)

    async def run_pipeline_async(self, max_books: int = None, output_format: str = "jsonl") -> str:
        """Async version of pipeline run with parallel book processing.

        Args:
            max_books: Maximum number of books to process.
            output_format: Output format ("jsonl" or "json").

        Returns:
            Path to the output file.
        """
        logger.info("Starting parallel pipeline run")

        # Get list of book files
        book_files = list(self.books_dir.glob("*.txt"))

        if max_books:
            book_files = book_files[:max_books]

        logger.info(f"Found {len(book_files)} books to process")

        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"longpage_dataset_parallel_{timestamp}.{output_format}"

        # Process books in parallel if enabled
        if self.enable_parallel and len(book_files) > 1:
            processed_books = await self._process_books_parallel(book_files)
        else:
            # Sequential processing
            processed_books = []
            for book_file in book_files:
                try:
                    book_data = self.process_book(str(book_file))
                    processed_books.append(book_data)
                except Exception as e:
                    logger.error(f"Failed to process {book_file.name}: {e}")
                    continue

        # Write output
        await self._write_output_async(processed_books, output_file, output_format)

        logger.info(f"Parallel pipeline complete!")
        logger.info(f"Processed: {len(processed_books)} books")
        logger.info(f"Output saved to: {output_file}")

        return str(output_file)

    async def _process_books_parallel(self, book_files: List[Path]) -> List[Dict[str, Any]]:
        """Process multiple books in parallel.

        Args:
            book_files: List of book file paths.

        Returns:
            List of processed book data.
        """
        logger.info(f"Processing {len(book_files)} books in parallel (max {self.max_concurrent_books} concurrent)")

        # Prepare book tasks
        book_tasks = [
            {"book_path": str(book_file), "index": i}
            for i, book_file in enumerate(book_files)
        ]

        async def process_single_book_task(book_task: Dict) -> Dict[str, Any]:
            """Process a single book task."""
            try:
                return await self.process_book_async(book_task["book_path"])
            except Exception as e:
                logger.error(f"Failed to process book {book_task['book_path']}: {e}")
                raise

        # Process books using parallel processor
        processed_books = await parallel_processor.process_books_parallel(
            book_tasks,
            process_single_book_task
        )

        return processed_books

    async def _write_output_async(self, books: List[Dict[str, Any]], output_file: Path, output_format: str):
        """Async write output to file.

        Args:
            books: List of processed book data.
            output_file: Output file path.
            output_format: Output format ("jsonl" or "json").
        """
        # Use thread executor for file I/O
        def write_sync():
            with open(output_file, 'w', encoding='utf-8') as f:
                if output_format == "jsonl":
                    for book_data in books:
                        f.write(json.dumps(book_data, ensure_ascii=False) + "\n")
                else:  # json
                    f.write("[\n")
                    for i, book_data in enumerate(books):
                        if i > 0:
                            f.write(",\n")
                        json.dump(book_data, f, ensure_ascii=False, indent=2)
                    f.write("\n]")

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, write_sync)

    def run_pipeline(self, max_books: int = None, output_format: str = "jsonl") -> str:
        """Run the pipeline with parallel processing.

        Args:
            max_books: Maximum number of books to process.
            output_format: Output format ("jsonl" or "json").

        Returns:
            Path to the output file.
        """
        if self.enable_parallel:
            return run_async(self.run_pipeline_async(max_books, output_format))
        else:
            # Fallback to original pipeline
            from pipeline import BookPipeline
            original_pipeline = BookPipeline(str(self.books_dir), str(self.output_dir))
            return original_pipeline.run_pipeline(max_books, output_format)

    def validate_output(self, output_file: str) -> Dict[str, Any]:
        """Validate the generated output file.

        Args:
            output_file: Path to the output file.

        Returns:
            Validation results.
        """
        # Import the original validation method
        from pipeline import BookPipeline
        original_pipeline = BookPipeline(str(self.books_dir), str(self.output_dir))
        return original_pipeline.validate_output(output_file)


def run_pipeline(max_books: int = None, output_format: str = "jsonl", enable_parallel: bool = True):
    """Convenience function to run the parallel pipeline.

    Args:
        max_books: Maximum number of books to process.
        output_format: Output format ("jsonl" or "json").
        enable_parallel: Whether to enable parallel processing.

    Returns:
        Path to the output file.
    """
    pipeline = ParallelBookPipeline(enable_parallel=enable_parallel)
    return pipeline.run_pipeline(max_books=max_books, output_format=output_format)


if __name__ == "__main__":
    # Run parallel pipeline with default settings
    output_file = run_pipeline(max_books=5, enable_parallel=True)
    print(f"Parallel pipeline completed. Output saved to: {output_file}")