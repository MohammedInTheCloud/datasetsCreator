"""
Scene segmentation agent for splitting chapters into scenes.
"""
import re
import logging
from typing import Dict, List, Tuple
import json

from agents.llm_agent import llm_agent

logger = logging.getLogger(__name__)


class SceneAgent:
    """Handles scene segmentation within chapters."""
    
    def __init__(self):
        """Initialize the scene agent."""
        # Scene boundary indicators
        self.scene_boundary_patterns = [
            # Time skips
            r'\b(later|earlier|the next day|the following morning|afterward|meanwhile)\b',
            # Location changes
            r'\b(arrived at|went to|came to|entered|left|departed from)\b',
            # Perspective changes
            r'\b(meanwhile|elsewhere|across town|in another part)\b',
            # Scene break markers
            r'\n\s*\*\s*\n|\n\s*-{3,}\s*\n|\n\s*\.\.\.\s*\n',
        ]
    
    def segment_scenes(self, chapter_title: str, chapter_text: str) -> List[Dict]:
        """Split a chapter into scenes using rule-based and LLM methods.
        
        Args:
            chapter_title: Title of the chapter.
            chapter_text: Full text of the chapter.
            
        Returns:
            List of scene dictionaries.
        """
        logger.info(f"Segmenting scenes for chapter: {chapter_title}")
        
        # First, try rule-based segmentation
        scenes = self._rule_based_segmentation(chapter_text)
        
        # Validate and refine with LLM
        if len(scenes) > 1:
            scenes = self._llm_validation(chapter_title, chapter_text, scenes)
        else:
            # If rule-based found nothing, use LLM-only segmentation
            scenes = self._llm_only_segmentation(chapter_title, chapter_text)
        
        logger.info(f"Found {len(scenes)} scenes in {chapter_title}")
        return scenes
    
    def _rule_based_segmentation(self, chapter_text: str) -> List[Dict]:
        """Attempt scene segmentation using rule-based methods.
        
        Args:
            chapter_text: Full chapter text.
            
        Returns:
            List of scene dictionaries.
        """
        scenes = []
        paragraphs = chapter_text.split('\n\n')
        
        current_scene = {
            "text": "",
            "paragraphs": []
        }
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if this paragraph indicates a scene boundary
            is_boundary = False
            for pattern in self.scene_boundary_patterns:
                if re.search(pattern, para, re.IGNORECASE):
                    is_boundary = True
                    break
            
            # If it's a boundary and we have content, finalize current scene
            if is_boundary and current_scene["text"]:
                scenes.append({
                    "text": current_scene["text"].strip(),
                    "paragraph_count": len(current_scene["paragraphs"])
                })
                current_scene = {"text": "", "paragraphs": []}
            
            # Add paragraph to current scene
            current_scene["text"] += para + "\n\n"
            current_scene["paragraphs"].append(para)
        
        # Add the last scene if it has content
        if current_scene["text"].strip():
            scenes.append({
                "text": current_scene["text"].strip(),
                "paragraph_count": len(current_scene["paragraphs"])
            })
        
        # If we only got one scene, try splitting by paragraph clusters
        if len(scenes) == 1 and len(paragraphs) > 10:
            scenes = self._paragraph_cluster_segmentation(chapter_text)
        
        return scenes
    
    def _paragraph_cluster_segmentation(self, chapter_text: str) -> List[Dict]:
        """Split chapter into scenes based on paragraph clusters.
        
        Args:
            chapter_text: Full chapter text.
            
        Returns:
            List of scene dictionaries.
        """
        paragraphs = [p.strip() for p in chapter_text.split('\n\n') if p.strip()]
        
        # Group paragraphs into scenes (aim for 3-7 paragraphs per scene)
        scenes = []
        paragraphs_per_scene = max(3, len(paragraphs) // 5)  # Target ~5 scenes
        
        for i in range(0, len(paragraphs), paragraphs_per_scene):
            scene_paragraphs = paragraphs[i:i + paragraphs_per_scene]
            scene_text = "\n\n".join(scene_paragraphs)
            
            scenes.append({
                "text": scene_text,
                "paragraph_count": len(scene_paragraphs)
            })
        
        return scenes
    
    def _llm_validation(self, chapter_title: str, chapter_text: str, rule_scenes: List[Dict]) -> List[Dict]:
        """Use LLM to validate and refine rule-based scene segmentation.
        
        Args:
            chapter_title: Title of the chapter.
            chapter_text: Full chapter text.
            rule_scenes: Scenes from rule-based segmentation.
            
        Returns:
            Validated list of scene dictionaries.
        """
        # Prepare scene summaries for LLM
        scene_summaries = []
        for i, scene in enumerate(rule_scenes):
            summary = scene["text"][:200] + "..." if len(scene["text"]) > 200 else scene["text"]
            scene_summaries.append(f"Scene {i+1}: {summary}")
        
        scenes_text = "\n\n".join(scene_summaries)
        
        prompt = f"""
        You are an expert literary analyst. I've split a chapter into scenes using rules, and I need you to validate and refine the segmentation.
        
        Chapter Title: {chapter_title}
        
        Current Scene Segmentation:
        {scenes_text}
        
        Full Chapter Text:
        {chapter_text[:3000]}...
        
        Please:
        1. Review the scene boundaries
        2. Merge any scenes that should be together
        3. Split any scenes that contain multiple distinct scenes
        4. Ensure each scene has a clear focus
        
        Respond with a JSON object containing the refined scenes:
        {{
            "scenes": [
                {{
                    "scene_name": "Brief descriptive name",
                    "start_word": 0,
                    "end_word": 250,
                    "narrative_focus": "Main character or focus of scene",
                    "narrative_perspective": "Third-person limited (Character) or First-person or Omniscient"
                }}
            ]
        }}
        
        Ensure the word ranges cover the entire chapter without gaps.
        """
        
        try:
            result = llm_agent.query_json(prompt, max_tokens=8192)
            
            # Convert word ranges back to text
            words = chapter_text.split()
            validated_scenes = []
            
            for scene_info in result.get("scenes", []):
                start_word = max(0, scene_info["start_word"])
                end_word = min(len(words), scene_info["end_word"])
                
                scene_words = words[start_word:end_word]
                scene_text = " ".join(scene_words)
                
                validated_scenes.append({
                    "scene_name": scene_info["scene_name"],
                    "text": scene_text,
                    "start_word": start_word,
                    "end_word": end_word,
                    "narrative_focus": scene_info["narrative_focus"],
                    "narrative_perspective": scene_info["narrative_perspective"]
                })
            logger.info(f"LLM validated and refined to {len(validated_scenes)} scenes" , result)
            print( f"LLM validated and refined to {len(validated_scenes)} scenes" , result)
            return validated_scenes
            
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            # Fall back to rule-based scenes with generic names
            return [
                {
                    "scene_name": f"Scene {i+1}",
                    "text": scene["text"],
                    "narrative_focus": "Unknown",
                    "narrative_perspective": "Unknown"
                }
                for i, scene in enumerate(rule_scenes)
            ]
    
    def _llm_only_segmentation(self, chapter_title: str, chapter_text: str) -> List[Dict]:
        """Use LLM to perform scene segmentation when rules fail.
        
        Args:
            chapter_title: Title of the chapter.
            chapter_text: Full chapter text.
            
        Returns:
            List of scene dictionaries.
        """
        prompt = f"""
        You are an expert literary analyst. Please identify scene boundaries within the following chapter.
        
        A scene typically changes when there is a significant shift in:
        - Time (e.g., "The next morning...")
        - Location (e.g., "Meanwhile, across town...")
        - Point of view or focus character
        - Major action or event
        
        Chapter Title: {chapter_title}
        
        Chapter Text (truncated if long):
        {chapter_text[:4000]}...
        
        Please identify 3-7 scenes and respond with:
        {{
            "scenes": [
                {{
                    "scene_name": "Brief descriptive name",
                    "start_word": 0,
                    "end_word": 500,
                    "narrative_focus": "Main character or focus",
                    "narrative_perspective": "Third-person limited or First-person"
                }}
            ]
        }}
        """
        
        try:
            result = llm_agent.query_json(prompt, max_tokens=8192)
            
            # Convert word ranges to text
            words = chapter_text.split()
            scenes = []
            
            for scene_info in result.get("scenes", []):
                start_word = max(0, scene_info["start_word"])
                end_word = min(len(words), scene_info["end_word"])
                
                scene_words = words[start_word:end_word]
                scene_text = " ".join(scene_words)
                
                scenes.append({
                    "scene_name": scene_info["scene_name"],
                    "text": scene_text,
                    "start_word": start_word,
                    "end_word": end_word,
                    "narrative_focus": scene_info["narrative_focus"],
                    "narrative_perspective": scene_info["narrative_perspective"]
                })
            
            return scenes
            
        except Exception as e:
            logger.error(f"LLM-only segmentation failed: {e}")
            # Ultimate fallback: split into equal parts
            return self._fallback_segmentation(chapter_text)
    
    def _fallback_segmentation(self, chapter_text: str) -> List[Dict]:
        """Ultimate fallback: equal segmentation.
        
        Args:
            chapter_text: Chapter text to segment.
            
        Returns:
            List of scene dictionaries.
        """
        words = chapter_text.split()
        target_scenes = 3
        words_per_scene = len(words) // target_scenes
        
        scenes = []
        for i in range(target_scenes):
            start = i * words_per_scene
            end = (i + 1) * words_per_scene if i < target_scenes - 1 else len(words)
            
            scene_words = words[start:end]
            scene_text = " ".join(scene_words)
            
            scenes.append({
                "scene_name": f"Scene {i+1}",
                "text": scene_text,
                "narrative_focus": "Unknown",
                "narrative_perspective": "Unknown"
            })
        
        return scenes


# Singleton instance
scene_agent = SceneAgent()