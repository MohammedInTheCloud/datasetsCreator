"""
Scene analysis agent for generating embeddings and summaries.
"""
import logging
from typing import Dict, List
import statistics
import os

from agents.llm_agent import llm_agent

logger = logging.getLogger(__name__)


class SceneAnalysisAgent:
    """Handles analysis of individual scenes."""
    
    def __init__(self):
        """Initialize the scene analysis agent."""
        # Dimensions for embedding space (configurable via env)
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
    
    def analyze_scene(self, scene_text: str, scene_name: str, 
                     narrative_focus: str = "", narrative_perspective: str = "") -> Dict:
        """Analyze a single scene and generate embeddings and summary.
        
        Args:
            scene_text: Text of the scene.
            scene_name: Name of the scene.
            narrative_focus: Main focus character(s) of the scene.
            narrative_perspective: Narrative perspective of the scene.
            
        Returns:
            Dictionary with scene analysis results.
        """
        logger.debug(f"Analyzing scene: {scene_name}")
        
        # Generate embedding space
        embedding_space = self._generate_embedding_space(scene_text, scene_name)
        
        # Generate scene summary
        scene_summary = self._generate_scene_summary(scene_text, scene_name)
        
        return {
            "scene_name": scene_name,
            "word_count": len(scene_text.split()),
            "embedding_space": embedding_space,
            "narrative_focus": narrative_focus,
            "narrative_perspective": narrative_perspective,
            "scene_summary_short": [scene_summary]
        }
    
    def _generate_embedding_space(self, scene_text: str, scene_name: str) -> Dict[str, int]:
        """Generate embedding space by averaging 16 LLM calls per dimension.
        
        Args:
            scene_text: Text of the scene.
            scene_name: Name of the scene.
            
        Returns:
            Dictionary with dimension scores.
        """
        embedding = {}
        
        # Truncate very long scenes
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
                    response = llm_agent.query(prompt, max_tokens=8192, temperature=0.1)
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
    
    def _parse_score(self, response: str) -> int:
        """Parse a score from LLM response.
        
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
    
    def _generate_scene_summary(self, scene_text: str, scene_name: str) -> str:
        """Generate a concise bullet-point summary of the scene.
        
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
            response = llm_agent.query(prompt, max_tokens=8192, temperature=0.3)
            # Clean up the response
            summary = response.strip().removeprefix("- ").removeprefix("• ").strip()
            return summary
        except Exception as e:
            logger.error(f"Failed to generate scene summary: {e}")
            return f"Scene events in {scene_name}"
    
    def aggregate_scene_embeddings(self, scenes: List[Dict]) -> Dict[str, int]:
        """Aggregate scene embeddings into chapter-level embeddings.
        
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


# Singleton instance
scene_analysis_agent = SceneAnalysisAgent()