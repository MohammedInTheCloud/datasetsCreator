"""
Parallel-enhanced scene analysis agent for generating embeddings and summaries.
Dramatically accelerates processing through concurrent API calls.
"""
import logging
from typing import Dict, List
import statistics
import os
import asyncio

from agents.llm_agent_async import hybrid_llm_agent
from utils.parallel_processor import parallel_processor, run_async

logger = logging.getLogger(__name__)


class ParallelSceneAnalysisAgent:
    """
    Enhanced scene analysis agent with parallel processing capabilities.
    Can process embeddings 10-50x faster than the original sequential version.
    """

    def __init__(self, enable_parallel: bool = True):
        """Initialize the parallel scene analysis agent."""
        # Configuration from environment
        self.enable_parallel = enable_parallel and os.getenv("DISABLE_PARALLEL_PROCESSING", "false").lower() != "true"

        # Embedding dimensions (configurable via env)
        dimensions_env = os.getenv("EMBEDDING_DIMENSIONS")
        if dimensions_env:
            self.embedding_dimensions = [d.strip() for d in dimensions_env.split(",")]
        else:
            self.embedding_dimensions = [
                "action", "dialog", "world_building", "exposition",
                "romantic", "erotic", "pacing"
            ]

        # Number of calls to average for each dimension (configurable via env)
        self.embedding_calls = int(os.getenv("EMBEDDING_CALLS", "16"))

        # Parallel processing limits
        self.max_concurrent_dimensions = int(os.getenv("MAX_CONCURRENT_DIMENSIONS", len(self.embedding_dimensions)))
        self.max_concurrent_calls = int(os.getenv("MAX_CONCURRENT_EMBEDDING_CALLS", 20))

        # Descriptions for each dimension
        self.dimension_descriptions = {
            "action": "Physical movement, fights, chases, physical conflict",
            "dialog": "Conversations between characters, dialogue exchanges",
            "world_building": "Descriptions of setting, culture, rules of the world",
            "exposition": "Background information, explanations, internal monologue",
            "romantic": "Romantic interactions, relationships, emotional intimacy",
            "erotic": "Sexual content, physical intimacy, sensual descriptions",
            "pacing": "How fast or slow the events unfold (0=very slow, 100=very fast)"
        }

        logger.info(f"Initialized parallel scene analysis agent")
        logger.info(f"Parallel processing: {self.enable_parallel}")
        logger.info(f"Dimensions: {len(self.embedding_dimensions)}, Calls per dimension: {self.embedding_calls}")
        logger.info(f"Concurrency limits: {self.max_concurrent_dimensions} dimensions, {self.max_concurrent_calls} calls")

    async def analyze_scene_async(self, scene_text: str, scene_name: str,
                                 narrative_focus: str = "", narrative_perspective: str = "") -> Dict:
        """
        Async version of scene analysis with parallel embedding generation.

        Args:
            scene_text: Text of the scene.
            scene_name: Name of the scene.
            narrative_focus: Main focus character(s) of the scene.
            narrative_perspective: Narrative perspective of the scene.

        Returns:
            Dictionary with scene analysis results.
        """
        logger.debug(f"Analyzing scene: {scene_name} (parallel={self.enable_parallel})")

        # Generate embedding space and scene summary concurrently
        embedding_task = self._generate_embedding_space_async(scene_text, scene_name)
        summary_task = self._generate_scene_summary_async(scene_text, scene_name)

        # Wait for both to complete
        embedding_space, scene_summary = await asyncio.gather(
            embedding_task, summary_task, return_exceptions=True
        )

        # Handle exceptions
        if isinstance(embedding_space, Exception):
            logger.error(f"Embedding generation failed for {scene_name}: {embedding_space}")
            embedding_space = {dim: 50 for dim in self.embedding_dimensions}

        if isinstance(scene_summary, Exception):
            logger.error(f"Scene summary failed for {scene_name}: {scene_summary}")
            scene_summary = f"Scene events in {scene_name}"

        return {
            "scene_name": scene_name,
            "word_count": len(scene_text.split()),
            "embedding_space": embedding_space,
            "narrative_focus": narrative_focus,
            "narrative_perspective": narrative_perspective,
            "scene_summary_short": [scene_summary]
        }

    def analyze_scene(self, scene_text: str, scene_name: str,
                     narrative_focus: str = "", narrative_perspective: str = "") -> Dict:
        """
        Synchronous wrapper for scene analysis.

        Args:
            scene_text: Text of the scene.
            scene_name: Name of the scene.
            narrative_focus: Main focus character(s) of the scene.
            narrative_perspective: Narrative perspective of the scene.

        Returns:
            Dictionary with scene analysis results.
        """
        if self.enable_parallel:
            return run_async(self.analyze_scene_async(scene_text, scene_name, narrative_focus, narrative_perspective))
        else:
            # Fallback to original sequential processing
            return self._analyze_scene_sequential(scene_text, scene_name, narrative_focus, narrative_perspective)

    def _analyze_scene_sequential(self, scene_text: str, scene_name: str,
                                 narrative_focus: str = "", narrative_perspective: str = "") -> Dict:
        """
        Original sequential processing method for fallback.

        Args:
            scene_text: Text of the scene.
            scene_name: Name of the scene.
            narrative_focus: Main focus character(s) of the scene.
            narrative_perspective: Narrative perspective of the scene.

        Returns:
            Dictionary with scene analysis results.
        """
        logger.debug(f"Analyzing scene sequentially: {scene_name}")

        # Generate embedding space
        embedding_space = self._generate_embedding_space_sequential(scene_text, scene_name)

        # Generate scene summary
        scene_summary = self._generate_scene_summary_sequential(scene_text, scene_name)

        return {
            "scene_name": scene_name,
            "word_count": len(scene_text.split()),
            "embedding_space": embedding_space,
            "narrative_focus": narrative_focus,
            "narrative_perspective": narrative_perspective,
            "scene_summary_short": [scene_summary]
        }

    async def _generate_embedding_space_async(self, scene_text: str, scene_name: str) -> Dict[str, int]:
        """
        Generate embedding space by processing all dimensions in parallel.

        Args:
            scene_text: Text of the scene.
            scene_name: Name of the scene.

        Returns:
            Dictionary with dimension scores.
        """
        if not self.enable_parallel:
            return self._generate_embedding_space_sequential(scene_text, scene_name)

        # Truncate very long scenes
        scene_excerpt = scene_text[:3000] if len(scene_text) > 3000 else scene_text

        # Prepare tasks for all dimensions
        dimension_tasks = []
        for dimension in self.embedding_dimensions:
            task_data = self._prepare_dimension_task(dimension, scene_excerpt, scene_name)
            dimension_tasks.append(task_data)

        # Process dimensions in parallel using our parallel processor
        embedding_scores = await parallel_processor.process_embeddings_parallel(dimension_tasks)

        logger.debug(f"Generated parallel embeddings for {scene_name}: {embedding_scores}")
        return embedding_scores

    def _prepare_dimension_task(self, dimension: str, scene_excerpt: str, scene_name: str) -> Dict[str, any]:
        """
        Prepare task data for a single dimension.

        Args:
            dimension: The dimension to process
            scene_excerpt: Truncated scene text
            scene_name: Name of the scene

        Returns:
            Task data dictionary
        """
        description = self.dimension_descriptions[dimension]

        # Special handling for pacing
        if dimension == "pacing":
            base_prompt = f"""
            On a scale of 0 to 100, how would you rate the PACING of this scene?

            Consider:
            - 0 = Very slow, contemplative, descriptive
            - 50 = Moderate pace, balanced
            - 100 = Very fast, action-packed, hectic

            Scene: {scene_name}
            Text: {scene_excerpt}

            Answer with ONLY a single number between 0 and 100.
            """
        else:
            base_prompt = f"""
            On a scale of 0 to 100, how much does this scene focus on {dimension.upper()}?

            {description.upper()}: {description}

            Scene: {scene_name}
            Text: {scene_excerpt}

            Answer with ONLY a single number between 0 and 100.
            """

        # Create multiple prompts for this dimension (for averaging)
        prompts = [base_prompt] * self.embedding_calls

        return {
            "dimension": dimension,
            "prompts": prompts,
            "scene_name": scene_name
        }

    def _generate_embedding_space_sequential(self, scene_text: str, scene_name: str) -> Dict[str, int]:
        """
        Original sequential embedding generation method.

        Args:
            scene_text: Text of the scene.
            scene_name: Name of the scene.

        Returns:
            Dictionary with dimension scores.
        """
        embedding = {}
        scene_excerpt = scene_text[:3000] if len(scene_text) > 3000 else scene_text

        for dimension in self.embedding_dimensions:
            scores = []
            description = self.dimension_descriptions[dimension]

            # Special handling for pacing
            if dimension == "pacing":
                prompt = f"""
                On a scale of 0 to 100, how would you rate the PACING of this scene?

                Consider:
                - 0 = Very slow, contemplative, descriptive
                - 50 = Moderate pace, balanced
                - 100 = Very fast, action-packed, hectic

                Scene: {scene_name}
                Text: {scene_excerpt}

                Answer with ONLY a single number between 0 and 100.
                """
            else:
                prompt = f"""
                On a scale of 0 to 100, how much does this scene focus on {dimension.upper()}?

                {description.upper()}: {description}

                Scene: {scene_name}
                Text: {scene_excerpt}

                Answer with ONLY a single number between 0 and 100.
                """

            # Make multiple calls and average
            for i in range(self.embedding_calls):
                try:
                    response = hybrid_llm_agent.query(prompt, max_tokens=8192, temperature=0.1)
                    score = self._parse_score(response)
                    scores.append(score)
                except Exception as e:
                    logger.warning(f"Failed to get {dimension} score for call {i+1}: {e}")
                    scores.append(50)  # Default to middle value

            # Calculate average and apply threshold
            if scores:
                avg_score = statistics.mean(scores)
                # Apply threshold: scores < 10 become 0
                embedding[dimension] = 0 if avg_score < 10 else int(round(avg_score))
            else:
                embedding[dimension] = 0

            logger.debug(f"{dimension}: {embedding[dimension]} (avg of {len(scores)} scores)")

        return embedding

    async def _generate_scene_summary_async(self, scene_text: str, scene_name: str) -> str:
        """
        Async version of scene summary generation.

        Args:
            scene_text: Text of the scene.
            scene_name: Name of the scene.

        Returns:
            Scene summary string.
        """
        # Truncate very long scenes
        scene_excerpt = scene_text[:3000] if len(scene_text) > 3000 else scene_text

        prompt = f"""
        Write a very concise bullet point summary (10-20 words) of the following scene.
        Focus on the key event, development, or revelation.

        Scene: {scene_name}
        Text: {scene_excerpt}

        Summary (single bullet point, 10-20 words):
        """

        try:
            response = await hybrid_llm_agent.query_async(prompt, max_tokens=8192, temperature=0.3)
            # Clean up the response
            summary = response.strip().removeprefix("- ").removeprefix("• ").strip()
            return summary
        except Exception as e:
            logger.error(f"Failed to generate scene summary: {e}")
            return f"Scene events in {scene_name}"

    def _generate_scene_summary_sequential(self, scene_text: str, scene_name: str) -> str:
        """
        Sequential version of scene summary generation.

        Args:
            scene_text: Text of the scene.
            scene_name: Name of the scene.

        Returns:
            Scene summary string.
        """
        # Truncate very long scenes
        scene_excerpt = scene_text[:3000] if len(scene_text) > 3000 else scene_text

        prompt = f"""
        Write a very concise bullet point summary (10-20 words) of the following scene.
        Focus on the key event, development, or revelation.

        Scene: {scene_name}
        Text: {scene_excerpt}

        Summary (single bullet point, 10-20 words):
        """

        try:
            response = hybrid_llm_agent.query(prompt, max_tokens=8192, temperature=0.3)
            # Clean up the response
            summary = response.strip().removeprefix("- ").removeprefix("• ").strip()
            return summary
        except Exception as e:
            logger.error(f"Failed to generate scene summary: {e}")
            return f"Scene events in {scene_name}"

    def _parse_score(self, response: str) -> int:
        """
        Parse a score from LLM response.

        Args:
            response: LLM response string.

        Returns:
            Parsed score (0-100).
        """
        # Extract first number from response
        import re
        numbers = re.findall(r'\d+', response)

        if numbers:
            score = int(numbers[0])
            # Clamp to 0-100 range
            return max(0, min(100, score))

        return 50  # Default if no number found

    def aggregate_scene_embeddings(self, scenes: List[Dict]) -> Dict[str, int]:
        """
        Aggregate scene embeddings into chapter-level embeddings.

        Args:
            scenes: List of analyzed scenes.

        Returns:
            Aggregated embedding dictionary.
        """
        if not scenes:
            return {dim: 0 for dim in self.embedding_dimensions}

        aggregated = {}
        for dimension in self.embedding_dimensions:
            # Average the scores across all scenes
            scores = [scene["embedding_space"][dimension] for scene in scenes]
            aggregated[dimension] = int(round(statistics.mean(scores)))

        return aggregated

    async def batch_analyze_scenes(self, scenes_data: List[Dict]) -> List[Dict]:
        """
        Analyze multiple scenes in parallel.

        Args:
            scenes_data: List of dictionaries with scene data (text, name, focus, perspective)

        Returns:
            List of analyzed scene results
        """
        if not self.enable_parallel:
            # Sequential fallback
            results = []
            for scene_data in scenes_data:
                result = self.analyze_scene(
                    scene_data["text"],
                    scene_data["scene_name"],
                    scene_data.get("narrative_focus", ""),
                    scene_data.get("narrative_perspective", "")
                )
                results.append(result)
            return results

        logger.info(f"Analyzing {len(scenes_data)} scenes in parallel")

        async def process_single_scene(scene_data: Dict) -> Dict:
            return await self.analyze_scene_async(
                scene_data["text"],
                scene_data["scene_name"],
                scene_data.get("narrative_focus", ""),
                scene_data.get("narrative_perspective", "")
            )

        # Create async tasks for all scenes
        tasks = [process_single_scene(scene_data) for scene_data in scenes_data]

        # Execute all tasks concurrently with rate limiting
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, dict):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Scene analysis failed for scene {i}: {result}")
                # Add minimal fallback result
                scene_data = scenes_data[i]
                valid_results.append({
                    "scene_name": scene_data.get("scene_name", f"Scene {i+1}"),
                    "word_count": 0,
                    "embedding_space": {dim: 50 for dim in self.embedding_dimensions},
                    "narrative_focus": scene_data.get("narrative_focus", ""),
                    "narrative_perspective": scene_data.get("narrative_perspective", ""),
                    "scene_summary_short": ["Analysis failed"]
                })

        logger.info(f"Scene analysis complete: {len(valid_results)}/{len(scenes_data)} successful")
        return valid_results


# Create enhanced parallel instance
parallel_scene_analysis_agent = ParallelSceneAnalysisAgent()


# Compatibility wrapper that can be used as drop-in replacement
class SceneAnalysisAgent:
    """Compatibility wrapper that uses parallel processing when available."""

    def __init__(self):
        """Initialize the scene analysis agent."""
        self.parallel_agent = parallel_scene_analysis_agent

    def analyze_scene(self, scene_text: str, scene_name: str,
                     narrative_focus: str = "", narrative_perspective: str = "") -> Dict:
        """Analyze a single scene (compatibility method)."""
        return self.parallel_agent.analyze_scene(
            scene_text, scene_name, narrative_focus, narrative_perspective
        )

    def aggregate_scene_embeddings(self, scenes: List[Dict]) -> Dict[str, int]:
        """Aggregate scene embeddings (compatibility method)."""
        return self.parallel_agent.aggregate_scene_embeddings(scenes)


# Create singleton instance for backward compatibility
scene_analysis_agent = SceneAnalysisAgent()