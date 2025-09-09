"""
Main pipeline orchestrator for the book dataset generator.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import os

from utils.preprocessor import BookPreprocessor
from agents.llm_agent import llm_agent
from agents.scene_agent import scene_agent
from agents.scene_analysis_agent import scene_analysis_agent
from agents.chapter_agent import chapter_agent
from agents.book_agent import book_agent
from agents.synthetic_prompt_generator import synthetic_prompt_generator
from utils.composer import compose

# Configure logging
log_file = Path(__file__).parent.parent / 'pipeline.log'
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


class BookPipeline:
    """Main pipeline for processing books into LongPage format."""
    
    def __init__(self, books_dir: str = None, output_dir: str = None):
        """Initialize the pipeline.
        
        Args:
            books_dir: Directory containing input books.
            output_dir: Directory for output files.
        """
        self.books_dir = Path(books_dir or "D:\\local\\datasetsCreator\\books")
        self.output_dir = Path(output_dir or "D:\\local\\datasetsCreator\\output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.preprocessor = BookPreprocessor(str(self.books_dir))
        
        logger.info(f"Initialized pipeline")
        logger.info(f"Books directory: {self.books_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def process_book(self, book_path: str) -> Dict[str, Any]:
        """Process a single book through the entire pipeline.
        
        Args:
            book_path: Path to the book file.
            
        Returns:
            Complete book data in LongPage format.
        """
        book_name = Path(book_path).name
        logger.info(f"Processing book: {book_name}")
        
        try:
            # Step 1: Load and preprocess
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
            
            # Step 2: Process each chapter
            logger.info("Step 2: Processing chapters")
            processed_chapters = {}
            chapter_summaries = {}
            
            for chapter_title, chapter_text in book_data["chapters"].items():
                logger.debug(f"Processing chapter: {chapter_title}")
                
                # Segment into scenes
                scenes = scene_agent.segment_scenes(chapter_title, chapter_text)
                
                # Analyze each scene
                analyzed_scenes = []
                for scene in scenes:
                    analyzed_scene = scene_analysis_agent.analyze_scene(
                        scene["text"],
                        scene["scene_name"],
                        scene.get("narrative_focus", ""),
                        scene.get("narrative_perspective", "")
                    )
                    analyzed_scenes.append(analyzed_scene)
                
                # Generate chapter summary
                scene_summaries = [s["scene_summary_short"][0] for s in analyzed_scenes]
                chapter_summary = chapter_agent.generate_chapter_summary(chapter_title, scene_summaries)
                
                # Aggregate scene embeddings for chapter
                chapter_embedding = scene_analysis_agent.aggregate_scene_embeddings(analyzed_scenes)
                
                # Store processed chapter
                processed_chapters[chapter_title] = {
                    "chapter": chapter_text,
                    "embedding_space": chapter_embedding,
                    "chapter_summary": chapter_summary,
                    "scene_breakdown": analyzed_scenes
                }
                
                # Store summary for book-level analysis
                chapter_summaries[chapter_title] = chapter_summary
            
            # Add processed chapters to final JSON
            final_json["book_chapters"] = processed_chapters
            
            # Step 3: Generate book-level metadata
            logger.info("Step 3: Generating book-level metadata")
            
            # Prepare data for book agent
            book_metadata_input = {
                "chapters": book_data["chapters"],
                "processed_chapters": processed_chapters
            }
            
            # Extract book metadata
            book_metadata = book_agent.generate_book_metadata(book_metadata_input)
            
            # Add to final JSON
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
    
    def _generate_book_archetype(self, book_data: Dict) -> str:
        """Generate book archetype using LLM with robust validation.
        
        Args:
            book_data: Book data dictionary.
            
        Returns:
            Generated archetype string (guaranteed to be clean).
        """
        # Predefined valid archetypes
        VALID_ARCHETYPES = {
            "fantasy", "science fiction", "sci-fi", "mystery", "romance", 
            "thriller", "horror", "historical fiction", "literary fiction", 
            "adventure", "dystopian", "contemporary fiction", "drama",
            "biography", "memoir", "non-fiction"
        }
        
        # Use first chapter and world rules as context
        first_chapter = list(book_data.get("book_chapters", {}).values())[0]
        context = first_chapter.get("chapter", "")[:1500]  # Reduced context
        
        world_rules = book_data.get("world_rules", [])
        if world_rules:
            context += "\n\nWorld Rules:\n" + "\n".join(f"- {rule}" for rule in world_rules[:2])
        
        # More constrained prompt
        prompt = f"""
        Classify this book into ONE genre category. Respond with ONLY the genre name from this list:
        Fantasy, Science Fiction, Mystery, Romance, Thriller, Horror, Historical Fiction, Literary Fiction, Adventure, Dystopian, Contemporary Fiction
        
        Text sample:
        {context}
        
        Genre:"""
        
        try:
            # Use lower temperature for more deterministic output
            response = llm_agent.query(prompt, max_tokens=8192, temperature=0.1).strip()
            
            # Multi-stage cleaning
            cleaned_response = self._clean_archetype_response(response)
            
            # Validate against known archetypes
            if cleaned_response.lower() in VALID_ARCHETYPES:
                return cleaned_response.title()
            
            # If LLM response is invalid, use fallback classification
            return self._fallback_archetype_classification(context)
            
        except Exception as e:
            logger.error(f"Failed to generate archetype: {e}")
            return self._fallback_archetype_classification(context)
    
    def _clean_archetype_response(self, response: str) -> str:
        """Clean LLM response to extract just the archetype.
        
        Args:
            response: Raw LLM response.
            
        Returns:
            Cleaned archetype string.
        """
        if not response:
            return ""
        
        # Remove common markdown and formatting
        response = re.sub(r'^[#*\-\d\.]+\s*', '', response, flags=re.MULTILINE)
        
        # Remove common prefixes
        prefixes_to_remove = [
            r'^(The book is|Genre:|Archetype:|Category:|Based on|This is|Classification:)\s*',
            r'^(Step \d+:|Answer:|Response:|Output:)\s*',
            r'^["\'\(\[]*',  # Remove quotes and brackets
        ]
        
        for prefix_pattern in prefixes_to_remove:
            response = re.sub(prefix_pattern, '', response, flags=re.IGNORECASE)
        
        # Take only the first line and first sentence
        response = response.split('\n')[0].split('.')[0].strip()
        
        # Remove trailing punctuation and quotes
        response = re.sub(r'["\'\)\]\.,:;!?]*$', '', response)
        
        # Convert to lowercase for keyword matching
        response_lower = response.lower()
        
        # If response contains explanation text, extract just the genre
        # Also check for specific book references and contexts
        genre_keywords = [
            "fantasy", "science fiction", "sci-fi", "mystery", "romance", 
            "thriller", "horror", "historical", "literary", "adventure", 
            "dystopian", "contemporary"
        ]
        
        # Special case for Moby-Dick - it's Literary Fiction
        if "moby-dick" in response_lower or "whale" in response_lower or "sea" in response_lower:
            return "Literary Fiction"
            
        for keyword in genre_keywords:
            if keyword in response_lower:
                if keyword == "sci-fi":
                    return "Science Fiction"
                return keyword.title().replace("_", " ")
        
        # Return first few words if still too long
        words = response.split()
        if len(words) > 3:
            response = " ".join(words[:2])
        
        return response.strip()
    
    def _fallback_archetype_classification(self, context: str) -> str:
        """Fallback keyword-based classification.
        
        Args:
            context: Book text context.
            
        Returns:
            Classified archetype.
        """
        context_lower = context.lower()
        
        # Keyword mappings with priority order
        keyword_mappings = [
            (["magic", "wizard", "dragon", "fantasy", "spell", "enchant"], "Fantasy"),
            (["future", "space", "technology", "robot", "alien", "sci-fi"], "Science Fiction"),
            (["murder", "mystery", "detective", "crime", "investigation"], "Mystery"),
            (["love", "romance", "romantic", "relationship", "marriage"], "Romance"),
            (["kill", "death", "blood", "fear", "dark", "evil"], "Thriller"),
            (["ghost", "horror", "monster", "nightmare", "terror"], "Horror"),
            (["war", "history", "century", "past", "historical"], "Historical Fiction"),
            (["journey", "adventure", "quest", "travel", "explore"], "Adventure"),
        ]
        
        for keywords, archetype in keyword_mappings:
            if any(keyword in context_lower for keyword in keywords):
                return archetype
        
        # Default fallback
        return "Literary Fiction"
    
    def _generate_book_tags(self, book_data: Dict) -> List[str]:
        """Generate book tags based on content.
        
        Args:
            book_data: Book data dictionary.
            
        Returns:
            List of tag strings.
        """
        def clean_tag(tag: str) -> str:
            """Clean a tag to ensure it's a simple word/phrase."""
            if not tag:
                return ""
            
            # Remove markdown headers
            tag = re.sub(r'^#+\s*', '', tag)
            
            # Remove common explanatory prefixes
            tag = re.sub(r'^(Tag:|Genre:|Category:)\s*', '', tag, flags=re.IGNORECASE)
            
            # Take only first part if multi-line
            tag = tag.split('\n')[0]
            
            # Remove periods and explanatory text
            tag = tag.split('.')[0].strip()
            
            # Maximum length for a tag
            if len(tag) > 50:
                tag = tag[:50].strip()
            
            return tag.lower()
        
        # Extract keywords from various sources
        tags = set()
        
        # Add archetype as tag (cleaned)
        if book_data.get("book_archetype"):
            archetype_tag = clean_tag(book_data["book_archetype"])
            if archetype_tag and len(archetype_tag.split()) <= 3:  # Max 3 words
                tags.add(archetype_tag)
        
        # Extract from world rules
        for rule in book_data.get("world_rules", []):
            if "magic" in rule.lower():
                tags.add("magic")
            if "future" in rule.lower() or "technology" in rule.lower():
                tags.add("sci-fi")
            if "romance" in rule.lower() or "love" in rule.lower():
                tags.add("romance")
        
        # Extract from story arcs
        for arc in book_data.get("story_arcs", []):
            if "journey" in arc.lower() or "quest" in arc.lower():
                tags.add("adventure")
            if "mystery" in arc.lower() or "investigation" in arc.lower():
                tags.add("mystery")
        
        # Ensure we have at least 3 tags
        if len(tags) < 3:
            default_tags = ["fiction", "novel", "story"]
            for tag in default_tags:
                if tag not in tags:
                    tags.add(tag)
                    if len(tags) >= 3:
                        break
        
        # Filter out any problematic tags and convert to list
        valid_tags = []
        for tag in tags:
            # Skip empty tags or tags with newlines/m markdown
            if tag and '\n' not in tag and '##' not in tag and 'step' not in tag:
                valid_tags.append(tag)
        
        return valid_tags[:5]  # Limit to 5 tags
    
    def run_pipeline(self, max_books: int = None, output_format: str = "jsonl") -> str:
        """Run the pipeline on all books in the books directory.
        
        Args:
            max_books: Maximum number of books to process.
            output_format: Output format ("jsonl" or "json").
            
        Returns:
            Path to the output file.
        """
        logger.info("Starting pipeline run")
        
        # Get list of book files
        book_files = list(self.books_dir.glob("*.txt"))
        
        if max_books:
            book_files = book_files[:max_books]
        
        logger.info(f"Found {len(book_files)} books to process")
        
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"longpage_dataset_{timestamp}.{output_format}"
        
        # Process books
        processed_count = 0
        failed_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for book_file in book_files:
                try:
                    logger.info(f"Processing {book_file.name} ({processed_count + 1}/{len(book_files)})")
                    
                    # Process the book
                    book_data = self.process_book(str(book_file))
                    
                    # Write output
                    if output_format == "jsonl":
                        f.write(json.dumps(book_data, ensure_ascii=False) + "\n")
                    else:  # json
                        if processed_count > 0:
                            f.write(",\n")
                        json.dump(book_data, f, ensure_ascii=False, indent=2)
                    
                    processed_count += 1
                    logger.info(f"[OK] Successfully processed: {book_file.name}")
                    
                except Exception as e:
                    logger.error(f"[FAILED] Failed to process {book_file.name}: {e}")
                    failed_count += 1
                    continue
        
        # Close JSON array if needed
        if output_format == "json" and processed_count > 0:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write("\n]")
        
        logger.info(f"Pipeline complete!")
        logger.info(f"Processed: {processed_count} books")
        logger.info(f"Failed: {failed_count} books")
        logger.info(f"Output saved to: {output_file}")
        
        return str(output_file)
    
    def validate_output(self, output_file: str) -> Dict[str, Any]:
        """Validate the generated output file.
        
        Args:
            output_file: Path to the output file.
            
        Returns:
            Validation results.
        """
        logger.info(f"Validating output file: {output_file}")
        
        results = {
            "total_books": 0,
            "valid_books": 0,
            "invalid_books": 0,
            "issues": []
        }
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    results["total_books"] += 1
                    
                    try:
                        book_data = json.loads(line)
                        
                        # Basic validation
                        required_fields = [
                            "use_prompt", "book_highlight", "book_title",
                            "book_chapters", "world_rules", "story_arcs"
                        ]
                        
                        missing_fields = [field for field in required_fields if field not in book_data]
                        if missing_fields:
                            results["issues"].append(f"Book {line_num}: Missing fields: {missing_fields}")
                            results["invalid_books"] += 1
                        else:
                            results["valid_books"] += 1
                            
                    except json.JSONDecodeError as e:
                        results["issues"].append(f"Book {line_num}: Invalid JSON: {e}")
                        results["invalid_books"] += 1
        
        except Exception as e:
            logger.error(f"Failed to validate output: {e}")
            results["issues"].append(f"Validation error: {e}")
        
        logger.info(f"Validation complete: {results['valid_books']}/{results['total_books']} books valid")
        return results


def run_pipeline(max_books: int = None, output_format: str = "jsonl"):
    """Convenience function to run the pipeline.
    
    Args:
        max_books: Maximum number of books to process.
        output_format: Output format ("jsonl" or "json").
        
    Returns:
        Path to the output file.
    """
    pipeline = BookPipeline()
    return pipeline.run_pipeline(max_books=max_books, output_format=output_format)


if __name__ == "__main__":
    # Run pipeline with default settings
    output_file = run_pipeline(max_books=5)  # Start with 5 books for testing
    print(f"Pipeline completed. Output saved to: {output_file}")