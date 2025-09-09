"""
Chapter-level analysis agent for generating chapter summaries.
"""
import logging
from typing import Dict, List

from agents.llm_agent import llm_agent

logger = logging.getLogger(__name__)


class ChapterAgent:
    """Handles chapter-level analysis and summarization."""
    
    def __init__(self):
        """Initialize the chapter agent."""
        pass
    
    def generate_chapter_summary(self, chapter_title: str, scene_summaries: List[str]) -> List[str]:
        """Generate a chapter summary from scene summaries.
        
        Args:
            chapter_title: Title of the chapter.
            scene_summaries: List of scene summary strings.
            
        Returns:
            List of bullet points summarizing the chapter.
        """
        logger.debug(f"Generating chapter summary for: {chapter_title}")
        
        # Combine scene summaries
        scene_bullets = "\n".join([f"- {summary}" for summary in scene_summaries])
        
        prompt = f"""
        You are a literary analyst. Below are bullet-point summaries of individual scenes from a chapter.
        
        Synthesize these into a cohesive chapter summary using 3-7 bullet points.
        Each bullet point should be 10-20 words.
        Focus on:
        - Main plot progression
        - Character developments
        - Key themes and events
        - Important revelations or decisions
        
        Chapter Title: {chapter_title}
        
        Scene Summaries:
        {scene_bullets}
        
        Chapter Summary (3-7 bullet points, 10-20 words each):
        """
        
        try:
            response = llm_agent.query(prompt, max_tokens=8192, temperature=0.3)
            
            # Parse bullet points from response
            bullet_points = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    # Clean up the bullet point
                    bullet = line.lstrip('-*• ').strip()
                    if bullet:
                        bullet_points.append(bullet)
            
            # If no bullet points found, try to split by newlines
            if not bullet_points:
                bullet_points = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Ensure we have at least 3 bullet points
            if len(bullet_points) < 3:
                bullet_points.extend([""] * (3 - len(bullet_points)))
            
            logger.debug(f"Generated {len(bullet_points)} bullet points for chapter summary")
            return bullet_points[:7]  # Limit to 7 max
            
        except Exception as e:
            logger.error(f"Failed to generate chapter summary: {e}")
            # Fallback: use scene summaries directly
            return scene_summaries[:7]
    
    def extract_chapter_themes(self, chapter_title: str, chapter_text: str) -> List[str]:
        """Extract key themes from a chapter.
        
        Args:
            chapter_title: Title of the chapter.
            chapter_text: Full text of the chapter.
            
        Returns:
            List of theme strings.
        """
        # Truncate long chapters
        chapter_excerpt = chapter_text[:3000] if len(chapter_text) > 3000 else chapter_text
        
        prompt = f"""
        Identify the main themes in this chapter. Themes are recurring ideas, concepts, or topics.
        
        Chapter Title: {chapter_title}
        
        Chapter Text:
        {chapter_excerpt}
        
        List 3-5 key themes as short phrases (2-4 words each):
        """
        
        try:
            response = llm_agent.query(prompt, max_tokens=8192, temperature=0.3)
            
            # Parse themes from response
            themes = []
            for line in response.split('\n'):
                line = line.strip()
                # Clean up common list formats
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    theme = line.lstrip('-*• ').strip()
                elif line and line[0].isdigit() and ('.' in line or ')' in line):
                    theme = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                else:
                    theme = line
                
                if theme and len(theme.split()) <= 4:
                    themes.append(theme)
            
            return themes[:5]  # Limit to 5 themes
            
        except Exception as e:
            logger.error(f"Failed to extract chapter themes: {e}")
            return []
    
    def analyze_chapter_pacing(self, scene_embeddings: List[Dict]) -> Dict:
        """Analyze the pacing of a chapter based on scene embeddings.
        
        Args:
            scene_embeddings: List of scene embedding dictionaries.
            
        Returns:
            Dictionary with pacing analysis.
        """
        if not scene_embeddings:
            return {"average_pacing": 50, "pacing_variance": 0, "pacing_trend": "stable"}
        
        pacing_scores = [scene["embedding_space"]["pacing"] for scene in scene_embeddings]
        
        # Calculate statistics
        avg_pacing = sum(pacing_scores) / len(pacing_scores)
        
        # Calculate variance
        variance = sum((score - avg_pacing) ** 2 for score in pacing_scores) / len(pacing_scores)
        
        # Determine trend
        if len(pacing_scores) >= 3:
            first_third = sum(pacing_scores[:len(pacing_scores)//3]) / (len(pacing_scores)//3)
            last_third = sum(pacing_scores[-(len(pacing_scores)//3):]) / (len(pacing_scores)//3)
            
            if last_third > first_third + 10:
                trend = "accelerating"
            elif last_third < first_third - 10:
                trend = "decelerating"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "average_pacing": int(round(avg_pacing)),
            "pacing_variance": int(round(variance)),
            "pacing_trend": trend
        }


# Singleton instance
chapter_agent = ChapterAgent()