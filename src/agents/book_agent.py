"""
Book-level analysis agents for extracting high-level book information.
"""
import logging
from typing import Dict, List, Set
import re

from agents.llm_agent import llm_agent

logger = logging.getLogger(__name__)


class BookAgent:
    """Handles book-level analysis and metadata extraction."""
    
    def __init__(self):
        """Initialize the book agent."""
        pass
    
    def extract_world_rules(self, book_data: Dict) -> List[str]:
        """Extract fundamental world rules from the book.
        
        Args:
            book_data: Dictionary containing book chapters.
            
        Returns:
            List of world rule strings.
        """
        logger.info("Extracting world rules")
        
        # Use first 1-2 chapters for context
        chapters = list(book_data.get("chapters", {}).values())
        if not chapters:
            return []
        
        context = "\n".join(chapters[:2])
        context = context[:8000]  # Limit context length
        
        prompt = f"""
        Analyze the following text from a book. Identify the fundamental "rules" of its world that differ from our modern real world.
        
        These could be:
        - Physical laws or magical systems
        - Social norms or cultural practices
        - Technological levels or limitations
        - Political structures or power systems
        - Supernatural elements or creatures
        
        List them as concise bullet points (10-20 words each). Focus only on rules that are explicitly shown or strongly implied.
        
        Text:
        {context}
        
        World Rules:
        """
        
        try:
            response = llm_agent.query(prompt, max_tokens=8192, temperature=0.2)
            
            # Parse bullet points - handle various formats
            rules = []
            for line in response.split('\n'):
                line = line.strip()
                
                # Skip empty lines and headers
                if not line or line.lower().startswith('world rules') or line.lower().startswith('here are'):
                    continue
                
                # Handle bullet points
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    rule = line.lstrip('-*• ').strip()
                    if rule:
                        rules.append(rule)
                # Handle numbered lists
                elif re.match(r'^\d+\.\s', line):
                    rule = re.sub(r'^\d+\.\s*', '', line)
                    if rule:
                        rules.append(rule)
            
            logger.info(f"Extracted {len(rules)} world rules")
            return rules
            
        except Exception as e:
            logger.error(f"Failed to extract world rules: {e}")
            return []
    
    def identify_story_arcs(self, chapter_summaries: Dict[str, List[str]]) -> List[str]:
        """Identify major story arcs throughout the book.
        
        Args:
            chapter_summaries: Dictionary mapping chapter titles to summaries.
            
        Returns:
            List of story arc descriptions.
        """
        logger.info("Identifying story arcs")
        
        # Combine all chapter summaries
        all_summaries = []
        for chapter_title, summaries in chapter_summaries.items():
            all_summaries.append(f"{chapter_title}:\n" + "\n".join(f"- {s}" for s in summaries))
        
        combined_summaries = "\n\n".join(all_summaries)
        
        prompt = f"""
        Analyze these chapter summaries to identify the major story arcs in this book.
        
        A story arc is a continuing storyline that spans multiple chapters, such as:
        - A character's journey or development
        - A romantic relationship
        - A mystery or investigation
        - A conflict or war
        - A quest or mission
        
        Chapter Summaries:
        {combined_summaries}
        
        List 3-7 major story arcs as concise descriptions (15-25 words each):
        """
        
        try:
            response = llm_agent.query(prompt, max_tokens=8192, temperature=0.3)
            
            # Parse story arcs - handle various formats
            arcs = []
            for line in response.split('\n'):
                line = line.strip()
                
                # Skip empty lines and headers
                if not line or line.lower().startswith('here are') or line.lower().startswith('major story'):
                    continue
                
                # Handle bullet points
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    arc = line.lstrip('-*• ').strip()
                    if arc:
                        arcs.append(arc)
                # Handle numbered lists (e.g., "1. ", "2. ")
                elif re.match(r'^\d+\.\s', line):
                    # Remove the number prefix
                    arc = re.sub(r'^\d+\.\s*', '', line)
                    if arc:
                        arcs.append(arc)
                # Handle lines that might be story arcs without prefixes
                elif len(line) > 20 and ':' in line:
                    # This might be a story arc description
                    arcs.append(line)
            
            logger.info(f"Identified {len(arcs)} story arcs")
            return arcs
            
        except Exception as e:
            logger.error(f"Failed to identify story arcs: {e}")
            return []
    
    def extract_characters(self, book_data: Dict) -> Dict[str, List[str]]:
        """Extract main and side characters from the book.
        
        Args:
            book_data: Dictionary containing book chapters.
            
        Returns:
            Dictionary with main_characters and side_characters lists.
        """
        logger.info("Extracting characters")
        
        # Use first 3 chapters and last 2 chapters for context
        chapters = list(book_data.get("chapters", {}).values())
        if len(chapters) < 2:
            return {"main_characters": [], "side_characters": []}
        
        # Sample chapters
        sample_chapters = chapters[:3] + chapters[-2:]
        context = "\n".join(sample_chapters)
        context = context[:10000]  # Limit context
        
        prompt = f"""
        Identify the characters in this book. Focus on named characters who appear multiple times or have significant impact.
        
        For each character, determine if they are a MAIN character or SIDE character:
        - MAIN: Protagonist, antagonist, love interest, or characters with major storylines
        - SIDE: Supporting characters, minor antagonists, characters who appear briefly
        
        Book Text (sample):
        {context}
        
        Respond in JSON format:
        {{
            "main_characters": ["Character Name 1", "Character Name 2"],
            "side_characters": ["Character Name 3", "Character Name 4"]
        }}
        """
        
        try:
            result = llm_agent.query_json(prompt, max_tokens=8192, temperature=0.2)
            
            main_chars = result.get("main_characters", [])
            side_chars = result.get("side_characters", [])
            
            logger.info(f"Extracted {len(main_chars)} main and {len(side_chars)} side characters")
            return {
                "main_characters": main_chars[:10],  # Limit to prevent too many
                "side_characters": side_chars[:20]
            }
            
        except Exception as e:
            logger.error(f"Failed to extract characters: {e}")
            return {"main_characters": [], "side_characters": []}
    
    def extract_character_archetypes(self, characters: Dict[str, List[str]], book_data: Dict) -> Dict[str, str]:
        """Extract character archetypes.
        
        Args:
            characters: Dictionary with main and side characters.
            book_data: Dictionary containing book chapters.
            
        Returns:
            Dictionary mapping character names to archetypes.
        """
        logger.info("Extracting character archetypes")
        
        all_characters = characters.get("main_characters", []) + characters.get("side_characters", [])
        if not all_characters:
            return {}
        
        # Get some context
        chapters = list(book_data.get("chapters", {}).values())
        context = "\n".join(chapters[:2])[:5000]
        
        prompt = f"""
        For each character listed, identify their literary archetype or role in the story.
        
        Common archetypes include: Hero, Mentor, Villain, Sidekick, Love Interest, Trickster, Outcast, etc.
        
        Characters: {', '.join(all_characters)}
        
        Book Context:
        {context}
        
        Respond in JSON format mapping character to archetype:
        {{
            "Character Name": "Archetype",
            "Another Character": "Different Archetype"
        }}
        """
        
        try:
            result = llm_agent.query_json(prompt, max_tokens=8192, temperature=0.2)
            
            # Clean up result
            archetypes = {}
            for char, archetype in result.items():
                if isinstance(archetype, str):
                    archetypes[char] = archetype
            
            logger.info(f"Extracted archetypes for {len(archetypes)} characters")
            return archetypes
            
        except Exception as e:
            logger.error(f"Failed to extract character archetypes: {e}")
            return {}
    
    def analyze_writing_style(self, book_data: Dict) -> List[str]:
        """Analyze the writing style of the book.
        
        Args:
            book_data: Dictionary containing book chapters.
            
        Returns:
            List of writing style descriptions.
        """
        logger.info("Analyzing writing style")
        
        # Sample from beginning, middle, and end
        chapters = list(book_data.get("chapters", {}).values())
        if len(chapters) < 3:
            return []
        
        samples = [
            chapters[0][:2000],  # Beginning
            chapters[len(chapters)//2][:2000],  # Middle
            chapters[-1][:2000]  # End
        ]
        
        context = "\n\n... [Chapter Break] ...\n\n".join(samples)
        
        prompt = f"""
        Analyze the writing style of this book based on these samples.
        
        Consider:
        - Narrative perspective (first-person, third-person limited, omniscient)
        - Tone and mood (serious, humorous, dark, light-hearted)
        - Prose style (descriptive, sparse, lyrical, straightforward)
        - Dialogue style (formal, casual, dialect-heavy)
        - Pacing and rhythm
        - Literary devices used
        
        Text Samples:
        {context}
        
        Describe the writing style in 3-5 bullet points (15-25 words each):
        """
        
        try:
            response = llm_agent.query(prompt, max_tokens=8192, temperature=0.3)
            
            # Parse bullet points - handle various formats
            styles = []
            for line in response.split('\n'):
                line = line.strip()
                
                # Skip empty lines and headers
                if not line or line.lower().startswith('writing style') or line.lower().startswith('here are'):
                    continue
                
                # Handle bullet points
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    style = line.lstrip('-*• ').strip()
                    if style:
                        styles.append(style)
                # Handle numbered lists
                elif re.match(r'^\d+\.\s', line):
                    style = re.sub(r'^\d+\.\s*', '', line)
                    if style:
                        styles.append(style)
            
            logger.info(f"Analyzed {len(styles)} aspects of writing style")
            return styles
            
        except Exception as e:
            logger.error(f"Failed to analyze writing style: {e}")
            return []
    
    def generate_book_metadata(self, book_data: Dict) -> Dict:
        """Generate all book-level metadata.
        
        Args:
            book_data: Dictionary containing book chapters and processed data.
            
        Returns:
            Dictionary with all book-level metadata.
        """
        logger.info("Generating book-level metadata")
        
        # Extract characters first
        characters = self.extract_characters(book_data)
        
        # Extract other metadata
        world_rules = self.extract_world_rules(book_data)
        
        # Get chapter summaries for story arcs
        chapter_summaries = {}
        for chapter_title, chapter_data in book_data.get("processed_chapters", {}).items():
            chapter_summaries[chapter_title] = chapter_data.get("chapter_summary", [])
        
        story_arcs = self.identify_story_arcs(chapter_summaries)
        
        character_archetypes = self.extract_character_archetypes(characters, book_data)
        writing_style = self.analyze_writing_style(book_data)
        
        return {
            "world_rules": world_rules,
            "story_arcs": story_arcs,
            "character_archetypes": character_archetypes,
            "writing_style": writing_style,
            "book_characters": characters
        }


# Singleton instance
book_agent = BookAgent()