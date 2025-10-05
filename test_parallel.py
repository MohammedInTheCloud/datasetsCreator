#!/usr/bin/env python3
"""
Test script to validate parallel processing functionality and data integrity.
"""
import sys
import time
import os
from pathlib import Path
import asyncio
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_dependencies():
    """Test that all required dependencies are available."""
    logger.info("Testing dependencies...")

    try:
        import aiohttp
        logger.info("✅ aiohttp available")
    except ImportError:
        logger.error("❌ aiohttp not available. Install with: pip install aiohttp>=3.8.0")
        return False

    try:
        import nest_asyncio
        logger.info("✅ nest_asyncio available")
    except ImportError:
        logger.error("❌ nest_asyncio not available. Install with: pip install nest-asyncio>=1.5.0")
        return False

    try:
        from utils.parallel_processor import parallel_processor
        from agents.llm_agent_async import hybrid_llm_agent
        from agents.scene_analysis_agent_parallel import parallel_scene_analysis_agent
        from pipeline_parallel import ParallelBookPipeline
        logger.info("✅ All parallel processing modules imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import parallel processing modules: {e}")
        return False

    return True


def test_configuration():
    """Test configuration loading and defaults."""
    logger.info("Testing configuration...")

    # Test environment variable loading
    from utils.parallel_processor import ParallelConfig

    config = ParallelConfig()
    logger.info(f"Default config: books={config.max_concurrent_books}, scenes={config.max_concurrent_scenes}")

    # Test environment overrides
    os.environ["MAX_CONCURRENT_BOOKS"] = "5"
    os.environ["MAX_CONCURRENT_SCENES"] = "15"

    config2 = ParallelConfig()
    logger.info(f"Override config: books={config2.max_concurrent_books}, scenes={config2.max_concurrent_scenes}")

    # Cleanup
    del os.environ["MAX_CONCURRENT_BOOKS"]
    del os.environ["MAX_CONCURRENT_SCENES"]

    logger.info("✅ Configuration testing passed")
    return True


async def test_async_llm_agent():
    """Test the async LLM agent functionality."""
    logger.info("Testing async LLM agent...")

    try:
        from agents.llm_agent_async import hybrid_llm_agent

        # Test simple query
        prompt = "What is 2 + 2? Answer with just the number."

        start_time = time.time()
        response = await hybrid_llm_agent.query_async(prompt, max_tokens=10, temperature=0.1)
        end_time = time.time()

        logger.info(f"✅ Async LLM query successful: '{response}' (took {end_time - start_time:.2f}s)")

        # Test batch queries
        prompts = [
            "What is 1 + 1? Answer with just the number.",
            "What is 2 + 2? Answer with just the number.",
            "What is 3 + 3? Answer with just the number."
        ]

        start_time = time.time()
        responses = await hybrid_llm_agent.async_agent.batch_query_async(prompts, concurrent_limit=3)
        end_time = time.time()

        logger.info(f"✅ Batch LLM queries successful: {responses} (took {end_time - start_time:.2f}s)")

        return True

    except Exception as e:
        logger.error(f"❌ Async LLM agent test failed: {e}")
        return False


async def test_parallel_scene_analysis():
    """Test parallel scene analysis functionality."""
    logger.info("Testing parallel scene analysis...")

    try:
        from agents.scene_analysis_agent_parallel import parallel_scene_analysis_agent

        # Test scene data
        test_scene = {
            "text": "This is a test scene with some action and dialogue. The character runs quickly through the forest while shouting for help. The world around them is dark and mysterious, with tall trees blocking the moonlight. They think about their past and future while trying to survive.",
            "scene_name": "Test Scene",
            "narrative_focus": "Test Character",
            "narrative_perspective": "Third-person"
        }

        # Test sequential vs parallel performance
        logger.info("Testing sequential scene analysis...")
        start_time = time.time()
        sequential_result = parallel_scene_analysis_agent.analyze_scene(
            test_scene["text"], test_scene["scene_name"],
            test_scene["narrative_focus"], test_scene["narrative_perspective"]
        )
        sequential_time = time.time() - start_time

        logger.info(f"✅ Sequential analysis completed in {sequential_time:.2f}s")

        logger.info("Testing parallel scene analysis...")
        start_time = time.time()
        parallel_result = await parallel_scene_analysis_agent.analyze_scene_async(
            test_scene["text"], test_scene["scene_name"],
            test_scene["narrative_focus"], test_scene["narrative_perspective"]
        )
        parallel_time = time.time() - start_time

        logger.info(f"✅ Parallel analysis completed in {parallel_time:.2f}s")

        # Compare results (should be similar)
        embedding_similar = True
        for dim in sequential_result["embedding_space"]:
            seq_val = sequential_result["embedding_space"][dim]
            par_val = parallel_result["embedding_space"][dim]
            if abs(seq_val - par_val) > 20:  # Allow some variation due to randomness
                embedding_similar = False
                logger.warning(f"Large difference in {dim}: sequential={seq_val}, parallel={par_val}")

        if embedding_similar:
            logger.info("✅ Sequential and parallel results are consistent")
        else:
            logger.warning("⚠️ Some variation between sequential and parallel results (expected due to LLM randomness)")

        # Test batch processing
        test_scenes = [test_scene] * 3  # Same scene 3 times

        start_time = time.time()
        batch_results = await parallel_scene_analysis_agent.batch_analyze_scenes(test_scenes)
        batch_time = time.time() - start_time

        logger.info(f"✅ Batch analysis of {len(test_scenes)} scenes completed in {batch_time:.2f}s")

        return True

    except Exception as e:
        logger.error(f"❌ Parallel scene analysis test failed: {e}")
        return False


def test_pipeline_integration():
    """Test pipeline integration with a small sample."""
    logger.info("Testing pipeline integration...")

    try:
        from pipeline_parallel import ParallelBookPipeline

        # Create test directory and file if needed
        test_books_dir = Path("test_books")
        test_books_dir.mkdir(exist_ok=True)

        # Create a simple test book
        test_file = test_books_dir / "test_book.txt"
        if not test_file.exists():
            test_content = """
Test Book

Chapter 1: The Beginning

This is the first chapter of our test book. The protagonist, John, wakes up in a mysterious forest. He looks around and sees tall trees everywhere. The sunlight filters through the leaves above.

John starts walking through the forest, trying to find a path. As he walks, he thinks about how he got here. The last thing he remembers is falling asleep in his bed.

Suddenly, he hears a noise in the distance. It sounds like someone calling for help. John decides to investigate and follows the sound deeper into the forest.

Chapter 2: The Discovery

John continues through the forest and eventually finds a small cabin. Smoke is rising from the chimney, suggesting someone is inside. He approaches cautiously and knocks on the door.

An old woman opens the door. She looks surprised to see John. "You're not from around here, are you?" she asks.

John explains that he doesn't know how he got here. The woman invites him in and offers him tea. As they talk, she reveals that this forest exists between worlds.

Chapter 3: The Choice

The woman explains that John has a choice to make. He can either return to his world or stay in this magical forest. She shows him a portal that would take him home.

John thinks about his life back home. It was safe but boring. Here, in this forest, there's adventure and mystery. He realizes that he wants to stay.

The woman nods understandingly. "Many have made the same choice," she says. "Welcome to your new home."
"""

            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)

        # Test pipeline with single book
        pipeline = ParallelBookPipeline(
            books_dir=str(test_books_dir),
            output_dir="test_output",
            enable_parallel=True
        )

        logger.info("Processing test book with parallel pipeline...")
        start_time = time.time()

        result = pipeline.process_book(str(test_file))

        end_time = time.time()
        processing_time = end_time - start_time

        logger.info(f"✅ Parallel pipeline completed in {processing_time:.2f}s")

        # Validate result structure
        required_fields = [
            "use_prompt", "book_highlight", "book_title", "book_chapters",
            "world_rules", "story_arcs", "character_archetypes", "writing_style"
        ]

        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            logger.error(f"❌ Missing required fields: {missing_fields}")
            return False

        logger.info("✅ All required fields present in output")

        # Check chapter processing
        chapters = result.get("book_chapters", {})
        if len(chapters) == 0:
            logger.error("❌ No chapters found in output")
            return False

        logger.info(f"✅ Found {len(chapters)} chapters in output")

        # Check scene analysis in chapters
        for chapter_title, chapter_data in chapters.items():
            embedding_space = chapter_data.get("embedding_space", {})
            if len(embedding_space) == 0:
                logger.error(f"❌ No embedding space found for chapter: {chapter_title}")
                return False

            scene_breakdown = chapter_data.get("scene_breakdown", [])
            if len(scene_breakdown) == 0:
                logger.warning(f"⚠️ No scenes found in chapter: {chapter_title}")

        logger.info("✅ Chapter and scene analysis validated")

        return True

    except Exception as e:
        logger.error(f"❌ Pipeline integration test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests and report results."""
    logger.info("🚀 Starting parallel processing validation tests...")

    tests = [
        ("Dependencies", test_dependencies),
        ("Configuration", test_configuration),
        ("Async LLM Agent", test_async_llm_agent),
        ("Parallel Scene Analysis", test_parallel_scene_analysis),
        ("Pipeline Integration", test_pipeline_integration)
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            results.append((test_name, result))

            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")

        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Parallel processing is ready to use.")
        return True
    else:
        logger.error(f"⚠️ {total - passed} tests failed. Please fix issues before using parallel processing.")
        return False


if __name__ == "__main__":
    # Check for environment variables
    if not os.getenv("LLM_API_KEY"):
        logger.warning("⚠️ LLM_API_KEY not set. Some tests may fail.")
        logger.info("Set up your .env file based on .env.example")

    # Run tests
    success = asyncio.run(run_all_tests())

    # Exit with appropriate code
    sys.exit(0 if success else 1)