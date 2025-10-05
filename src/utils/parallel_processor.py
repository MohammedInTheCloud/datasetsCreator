"""
Parallel processing utilities for the LongPage Dataset Generator.
Provides async capabilities with rate limiting and error handling.
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    max_concurrent_books: int = 3
    max_concurrent_scenes: int = 10
    max_concurrent_embeddings: int = 20
    rate_limit_delay: float = 0.1
    max_retries: int = 3
    enable_parallel: bool = True


class AsyncRateLimiter:
    """Async rate limiter using semaphore-based throttling."""

    def __init__(self, rate_limit: float, max_concurrent: int):
        """
        Initialize rate limiter.

        Args:
            rate_limit: Minimum delay between requests (seconds)
            max_concurrent: Maximum concurrent requests
        """
        self.rate_limit = rate_limit
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.last_request_time = 0

    async def acquire(self):
        """Acquire permission to make a request."""
        await self.semaphore.acquire()

        # Rate limiting delay
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)

        self.last_request_time = time.time()

    def release(self):
        """Release permission after request completes."""
        self.semaphore.release()


class ParallelProcessor:
    """Main parallel processing utility."""

    def __init__(self, config: Optional[ParallelConfig] = None):
        """Initialize parallel processor."""
        self.config = config or ParallelConfig()

        # Load from environment if available
        self.config.max_concurrent_books = int(os.getenv("MAX_CONCURRENT_BOOKS", self.config.max_concurrent_books))
        self.config.max_concurrent_scenes = int(os.getenv("MAX_CONCURRENT_SCENES", self.config.max_concurrent_scenes))
        self.config.max_concurrent_embeddings = int(os.getenv("MAX_CONCURRENT_EMBEDDINGS", self.config.max_concurrent_embeddings))
        self.config.rate_limit_delay = float(os.getenv("PARALLEL_RATE_LIMIT_DELAY", self.config.rate_limit_delay))

        # Create rate limiters
        self.embedding_limiter = AsyncRateLimiter(
            self.config.rate_limit_delay,
            self.config.max_concurrent_embeddings
        )
        self.scene_limiter = AsyncRateLimiter(
            self.config.rate_limit_delay * 2,  # Slightly slower for scene processing
            self.config.max_concurrent_scenes
        )

        logger.info(f"Initialized parallel processor with config: {self.config}")

    async def process_embeddings_parallel(
        self,
        embedding_tasks: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Process multiple embedding dimensions in parallel.

        Args:
            embedding_tasks: List of dictionaries with dimension info and prompts

        Returns:
            Dictionary mapping dimension names to averaged scores
        """
        if not self.config.enable_parallel:
            # Fallback to sequential processing
            return await self._process_embeddings_sequential(embedding_tasks)

        logger.debug(f"Processing {len(embedding_tasks)} embedding dimensions in parallel")

        async def process_single_dimension(task_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single dimension with multiple calls."""
            dimension = task_data["dimension"]
            prompts = task_data["prompts"]

            async with self.embedding_limiter:
                try:
                    scores = []
                    for prompt in prompts:
                        score = await self._make_llm_call_async(prompt)
                        scores.append(score)

                    avg_score = sum(scores) / len(scores) if scores else 50
                    return {"dimension": dimension, "score": int(round(avg_score))}

                except Exception as e:
                    logger.warning(f"Failed to process dimension {dimension}: {e}")
                    return {"dimension": dimension, "score": 50}  # Default fallback

        # Process all dimensions concurrently
        tasks = [process_single_dimension(task) for task in embedding_tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Extract successful results
        embedding_scores = {}
        for result in results:
            if isinstance(result, dict) and "dimension" in result:
                embedding_scores[result["dimension"]] = result["score"]
            elif isinstance(result, Exception):
                logger.error(f"Embedding processing failed: {result}")

        return embedding_scores

    async def _process_embeddings_sequential(self, embedding_tasks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Sequential fallback for embedding processing."""
        embedding_scores = {}

        for task_data in embedding_tasks:
            dimension = task_data["dimension"]
            prompts = task_data["prompts"]

            try:
                scores = []
                for prompt in prompts:
                    score = await self._make_llm_call_async(prompt)
                    scores.append(score)

                avg_score = sum(scores) / len(scores) if scores else 50
                embedding_scores[dimension] = int(round(avg_score))

            except Exception as e:
                logger.warning(f"Failed to process dimension {dimension}: {e}")
                embedding_scores[dimension] = 50

        return embedding_scores

    async def _make_llm_call_async(self, prompt: str) -> int:
        """
        Make an async LLM call and parse the numeric response.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Parsed numeric score (0-100)
        """
        # Import here to avoid circular imports
        from agents.llm_agent_async import hybrid_llm_agent

        try:
            response = await hybrid_llm_agent.query_async(prompt, max_tokens=8192, temperature=0.1)
            return self._parse_score(response)
        except Exception as e:
            logger.warning(f"Async LLM call failed: {e}")
            # Fallback to sync agent
            try:
                from agents.llm_agent import llm_agent
                response = llm_agent.query(prompt, max_tokens=8192, temperature=0.1)
                return self._parse_score(response)
            except Exception as e2:
                logger.warning(f"Sync LLM fallback also failed: {e2}")
                return 50  # Default fallback

    def _parse_score(self, response: str) -> int:
        """Parse a numeric score from LLM response."""
        import re
        numbers = re.findall(r'\d+', response)

        if numbers:
            score = int(numbers[0])
            return max(0, min(100, score))

        return 50

    async def process_scenes_parallel(
        self,
        scene_tasks: List[Dict[str, Any]],
        processor_func: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple scenes in parallel.

        Args:
            scene_tasks: List of scene data dictionaries
            processor_func: Async function to process each scene

        Returns:
            List of processed scene results
        """
        if not self.config.enable_parallel:
            # Sequential fallback
            results = []
            for task in scene_tasks:
                result = await processor_func(task)
                results.append(result)
            return results

        logger.debug(f"Processing {len(scene_tasks)} scenes in parallel")

        async def process_single_scene(scene_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single scene with rate limiting."""
            async with self.scene_limiter:
                try:
                    return await processor_func(scene_data)
                except Exception as e:
                    logger.error(f"Failed to process scene {scene_data.get('scene_name', 'unknown')}: {e}")
                    # Return minimal fallback data
                    return {
                        "scene_name": scene_data.get("scene_name", "unknown"),
                        "word_count": 0,
                        "embedding_space": {dim: 50 for dim in ["action", "dialog", "world_building", "exposition", "romantic", "erotic", "pacing"]},
                        "scene_summary_short": ["Scene processing failed"]
                    }

        # Process scenes concurrently
        tasks = [process_single_scene(scene_data) for scene_data in scene_tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, dict):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Scene processing exception: {result}")

        return valid_results

    async def process_books_parallel(
        self,
        book_tasks: List[Dict[str, Any]],
        processor_func: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple books in parallel.

        Args:
            book_tasks: List of book file paths or data
            processor_func: Async function to process each book

        Returns:
            List of processed book results
        """
        if not self.config.enable_parallel:
            # Sequential fallback
            results = []
            for task in book_tasks:
                result = await processor_func(task)
                results.append(result)
            return results

        logger.info(f"Processing {len(book_tasks)} books in parallel (max {self.config.max_concurrent_books} concurrent)")

        # Create semaphore for book-level concurrency
        book_semaphore = asyncio.Semaphore(self.config.max_concurrent_books)

        async def process_single_book(book_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single book with concurrency control."""
            async with book_semaphore:
                try:
                    return await processor_func(book_data)
                except Exception as e:
                    logger.error(f"Failed to process book {book_data.get('book_path', 'unknown')}: {e}")
                    raise

        # Process books concurrently with semaphore limiting
        tasks = [process_single_book(book_data) for book_data in book_tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successful and failed results
        successful_results = []
        failed_count = 0

        for result in results:
            if isinstance(result, dict):
                successful_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Book processing failed: {result}")
                failed_count += 1

        logger.info(f"Book processing complete: {len(successful_results)} successful, {failed_count} failed")
        return successful_results


# Global instance for easy import
parallel_processor = ParallelProcessor()


def run_async(coro):
    """
    Helper function to run async code from sync context.

    Args:
        coro: Async coroutine to run

    Returns:
        Result of the coroutine
    """
    try:
        # Try to get existing event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    # If we have a running loop, we can use nest_asyncio
    import nest_asyncio
    nest_asyncio.apply(loop)
    return loop.run_until_complete(coro)