"""
Data preprocessing module for loading and cleaning book texts.
"""
import re
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import unicodedata

from agents.llm_agent import llm_agent

logger = logging.getLogger(__name__)


class BookPreprocessor:
    """Handles loading, cleaning, and preprocessing of book texts."""
    
    def __init__(self, books_dir: str = "D:\\local\\tahdari\\FINALAPP\\ien\\rag\\books"):
        """Initialize the preprocessor.
        
        Args:
            books_dir: Directory containing book files.
        """
        self.books_dir = Path(books_dir)
        
    def load_and_clean_book(self, file_path: str) -> Dict:
        """Load a book file and clean its content.
        
        Args:
            file_path: Path to the book file.
            
        Returns:
            Dictionary containing cleaned book data.
        """
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            raw_text = f.read()
            
        # Clean the text
        cleaned_text = self._clean_text(raw_text)
        
        # Extract title from filename
        title = Path(file_path).stem.replace('_', ' ').title()
        
        return {
            "title": title,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean raw text content.
        
        Args:
            text: Raw text to clean.
            
        Returns:
            Cleaned text.
        """
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove Project Gutenberg headers/footers
        text = self._remove_gutenberg_boilerplate(text)
        
        # Fix common encoding issues
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive line breaks
        text = re.sub(r' {2,}', ' ', text)     # Remove multiple spaces
        
        # Remove weird quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '--')
        
        return text.strip()
    
    def _remove_gutenberg_boilerplate(self, text: str) -> str:
        """Remove Project Gutenberg header and footer.
        
        Args:
            text: Text with potential boilerplate.
            
        Returns:
            Text with boilerplate removed.
        """
        # Look for common Gutenberg patterns
        start_patterns = [
            r'\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK',
            r'Project Gutenberg\'s.*',
            r'The Project Gutenberg EBook of',
        ]
        
        end_patterns = [
            r'\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK',
            r'End of (the )?Project Gutenberg',
            r'End of.*Project Gutenberg',
        ]
        
        # Find start
        start_idx = 0
        for pattern in start_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Find the next empty line after the match
                next_newline = text.find('\n\n', match.end())
                if next_newline != -1:
                    start_idx = next_newline + 2
                break
        
        # Find end
        end_idx = len(text)
        for pattern in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Find the previous empty line before the match
                prev_newline = text.rfind('\n\n', 0, match.start())
                if prev_newline != -1:
                    end_idx = prev_newline
                break
        
        return text[start_idx:end_idx].strip()
    
    def _find_story_start(self, text: str) -> str:
        """Find where the actual story content begins, skipping table of contents.
        
        Args:
            text: Full book text.
            
        Returns:
            Text starting from where the actual story begins.
        """
        # First, look specifically for Chapter 1 with substantial content
        chapter1_pattern = re.compile(r'(?i)\n\s*(CHAPTER\s+1\b\.?\s*[^\n]*)')
        
        best_match = None
        max_content_length = 0
        
        # Find all Chapter 1 occurrences
        for match in chapter1_pattern.finditer(text):
            chapter_start = match.start()
            content_start = match.end()
            
            # Check content after this chapter
            next_chapter_pattern = re.compile(r'(?i)\n\s*(CHAPTER\s+[IVXLCDM\d]+\.?\s*[^\n]*)')
            next_chapter_match = next_chapter_pattern.search(text, content_start)
            
            if next_chapter_match:
                content_between = text[content_start:next_chapter_match.start()]
            else:
                content_between = text[content_start:content_start + 2000]  # Look at first 2000 chars
            
            content_between = content_between.strip()
            word_count = len(content_between.split())
            
            # Skip if this looks like TOC
            if self._is_toc_entry(content_between):
                continue
            
            # Prioritize chapters with more content
            if word_count > 100 and word_count > max_content_length:
                max_content_length = word_count
                best_match = match
        
        if best_match:
            logger.info(f"Found story start at Chapter 1, position {best_match.start()}: {best_match.group(1)}")
            return text[best_match.start():]
        
        # If no Chapter 1 found, try other patterns
        story_start_patterns = [
            r'(?i)\n\s*Call me Ishmael',  # Moby Dick specific
            r'(?i)\n\s*It was the best of times',  # Tale of Two Cities
            r'(?i)\n\s*Mr. Jones',  # Animal Farm
            r'(?i)\n\s*When Mr. Bilbo Baggins',  # The Hobbit
            r'(?i)\n\s*You will rejoice to hear',  # Frankenstein
            r'(?i)\n\s*The sun shone brightly',  # Common opening
        ]
        
        for pattern in story_start_patterns:
            match = re.search(pattern, text)
            if match:
                logger.info(f"Found story start using pattern: {pattern}")
                # Find the beginning of this line
                line_start = text.rfind('\n', 0, match.start()) + 1
                return text[line_start:]
        
        # Last resort: look for the first paragraph with substantial content
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line.strip() and len(line.strip()) > 50:  # Substantial line
                # Check if the next few lines also have content
                next_content = '\n'.join(lines[i:i+5])
                if len(next_content.split()) > 100:
                    logger.info(f"Found story start at line {i}")
                    return '\n'.join(lines[i:])
        
        logger.warning("Could not find story start, using full text")
        return text
    
    def _is_toc_entry(self, text: str) -> bool:
        """Check if text appears to be a table of contents entry.
        
        Args:
            text: Text to check.
            
        Returns:
            True if text appears to be TOC.
        """
        # TOC entries are typically short, just page numbers, or lists of chapters
        lines = text.split('\n')
        
        # If it's just a few short lines, likely TOC
        if len(lines) < 5:
            # Check for page numbers
            if re.search(r'\b\d+\s*$', text):  # Page numbers at end of lines
                return True
            
            # Check if it's just chapter titles without content
            non_empty_lines = [l for l in lines if l.strip()]
            if len(non_empty_lines) < 3:
                return True
        
        # Check for TOC patterns
        toc_patterns = [
            r'(?i)contents?',
            r'(?i)table of contents',
            r'(?i)chapter\s+[IVXLCDM\d]+\s*\.\s*\d+',  # Chapter with page number
            r'(?i)^\s*[IVXLCDM\d]+\s*\.\s*\d+\s*$',  # Roman numeral with page
        ]
        
        for pattern in toc_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def split_into_chapters(self, text: str, min_chapters: int = 4) -> Dict[str, str]:
        """Split text into chapters using improved rule-based method.
        
        Args:
            text: Full book text.
            min_chapters: Minimum number of chapters required.
            
        Returns:
            Dictionary mapping chapter titles to chapter text.
        """
        # First, find where the actual story content begins
        # Skip table of contents, preface, etc.
        text = self._find_story_start(text)
        
        # Use improved regex pattern that better handles chapter boundaries
        # This pattern matches chapter headings and captures only the title
        chapter_pattern = re.compile(
            r'(?i)(?:^|\n)(CHAPTER\s+[IVXLCDM\d]+(?:\.[^\n]*)?)\s*\n'
        )
        
        # Also try a simpler pattern for books with minimal chapter headings
        simple_chapter_pattern = re.compile(
            r'(?i)(?:^|\n)(CHAPTER\s+[IVXLCDM\d]+)\s*(?=\n|$)'
        )
        
        # Find all chapter matches
        matches = list(chapter_pattern.finditer(text))
        
        # If main pattern doesn't find enough chapters, try simple pattern
        if len(matches) < min_chapters:
            simple_matches = list(simple_chapter_pattern.finditer(text))
            if len(simple_matches) >= min_chapters:
                matches = simple_matches
                logger.info(f"Using simple chapter pattern, found {len(matches)} matches")
        
        chapters = {}
        
        if len(matches) >= min_chapters:
            logger.info(f"Found {len(matches)} chapters using improved pattern")
            
            # Extract chapters from matches
            for i, match in enumerate(matches):
                # Get chapter title (clean it up)
                title = match.group(1).strip()
                # Clean up the title - ensure it only contains the chapter heading
                title = re.sub(r'^CHAPTER\s+', 'Chapter ', title, flags=re.IGNORECASE)
                # Remove any trailing colons or extra punctuation
                title = re.sub(r'\s*:+\s*$', '', title)
                # Limit title length to prevent capturing content
                title = title[:500]  # Reasonable limit for chapter titles
                
                # Find chapter content
                start_pos = match.end()
                
                # Find end position (next chapter or end of text)
                if i < len(matches) - 1:
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(text)
                
                # Extract chapter content
                content = text[start_pos:end_pos].strip()
                
                # Remove title from content if it appears at the beginning
                content = re.sub(r'^' + re.escape(match.group(1).strip()) + r'\s*', '', content, count=1, flags=re.IGNORECASE)
                content = content.strip()
                
                if title and content and len(content.split()) > 50:  # Reduced minimum word count
                    chapters[title] = content
                else:
                    logger.debug(f"Skipping chapter '{title}': content length = {len(content.split()) if content else 0} words")
        
        logger.info(f"After filtering: {len(chapters)} chapters remain (min required: {min_chapters})")
        
        # If improved pattern failed, fall back to original patterns
        if not chapters or len(chapters) < min_chapters:
            logger.warning("Improved pattern failed, falling back to original patterns")
            return self._fallback_split_patterns(text, min_chapters)
        
        logger.info(f"Successfully extracted {len(chapters)} chapters")
        return chapters
    
    def _fallback_split_patterns(self, text: str, min_chapters: int) -> Dict[str, str]:
        """Fallback to original patterns if improved method fails.
        
        Args:
            text: Full book text.
            min_chapters: Minimum number of chapters required.
            
        Returns:
            Dictionary mapping chapter titles to chapter text.
        """
        # Common chapter patterns (original logic)
        patterns = [
            # Chapter with number
            (r'\n\s*(Chapter|CHAPTER)\s+[IVXLCDM0-9]+(?:\.\s+|\s*:\s*|\s+)([^\n]+)', 2),
            # Roman numerals alone
            (r'\n\s*([IVXLCDM]+)\s*\n', 1),
            # Just "Chapter" followed by text
            (r'\n\s*(Chapter|CHAPTER)\s*:\s*([^\n]+)', 2),
            # Numbered chapters
            (r'\n\s*\d+\.\s*([^\n]+)', 1),
        ]
        
        chapters = {}
        
        for pattern, title_group in patterns:
            matches = list(re.finditer(pattern, text))
            
            if len(matches) >= min_chapters:
                logger.info(f"Found {len(matches)} chapters using fallback pattern: {pattern}")
                
                # Extract chapters
                for i, match in enumerate(matches):
                    start_pos = match.start()
                    
                    # Get chapter title
                    if title_group == 1:
                        title = match.group(1).strip()
                    else:
                        title = match.group(2).strip()
                    
                    # If title is just a number/roman, make it more descriptive
                    if re.match(r'^[IVXLCDM0-9]+$', title):
                        title = f"Chapter {title}"
                    
                    # Clean up the title
                    title = re.sub(r'^CHAPTER\s+', 'Chapter ', title, flags=re.IGNORECASE)
                    title = re.sub(r'\s*:+\s*$', '', title)
                    title = title[:500]  # Reasonable limit for chapter titles
                    
                    # Find end position (next chapter or end of text)
                    if i < len(matches) - 1:
                        end_pos = matches[i + 1].start()
                    else:
                        end_pos = len(text)
                    
                    # Extract chapter text
                    chapter_text = text[start_pos:end_pos].strip()
                    
                    # Clean chapter text (remove the title line)
                    chapter_text = re.sub(r'^' + re.escape(match.group(0).strip()) + r'\s*', '', chapter_text, count=1)
                    chapter_text = chapter_text.strip()
                    
                    if chapter_text and len(chapter_text.split()) > 50:  # Reduced minimum word count
                        chapters[title] = chapter_text
                
                break
        
        return chapters
    
    def _llm_chapter_splitting(self, text: str, min_chapters: int) -> Dict[str, str]:
        """Use LLM to split text into chapters when rule-based methods fail.
        
        Args:
            text: Full book text.
            min_chapters: Minimum number of chapters.
            
        Returns:
            Dictionary mapping chapter titles to chapter text.
        """
        # Truncate text for LLM
        truncated_text = text[:10000]  # First 10k characters
        
        prompt = f"""
        You are an expert literary analyst. I need you to identify chapter boundaries in the following book text.
        
        Please identify at least {min_chapters} natural breaking points that could serve as chapter divisions.
        Look for:
        - Changes in time or location
        - Shifts in perspective
        - Natural pauses in the narrative
        - Section breaks marked by extra whitespace
        
        For each chapter you identify, provide:
        1. A descriptive title for the chapter
        2. The approximate starting word number (0-based index)
        
        Respond in valid JSON format like this:
        {{
            "chapters": [
                {{
                    "title": "Chapter 1: The Beginning",
                    "start_word": 0
                }},
                {{
                    "title": "Chapter 2: The Journey", 
                    "start_word": 1500
                }}
            ]
        }}
        
        Book text (first portion):
        {truncated_text}
        """
        
        try:
            result = llm_agent.query_json(prompt, max_tokens=1024)
            
            # Extract chapters based on word positions
            words = text.split()
            chapters = {}
            
            for i, chapter_info in enumerate(result.get("chapters", [])):
                title = chapter_info["title"]
                start_word = chapter_info["start_word"]
                
                # Find end word (next chapter or end)
                if i < len(result["chapters"]) - 1:
                    end_word = result["chapters"][i + 1]["start_word"]
                else:
                    end_word = len(words)
                
                # Extract chapter text
                chapter_words = words[start_word:end_word]
                chapter_text = " ".join(chapter_words)
                
                if chapter_text.strip():
                    chapters[title] = chapter_text
            
            logger.info(f"LLM identified {len(chapters)} chapters")
            return chapters
            
        except Exception as e:
            logger.error(f"LLM chapter splitting failed: {e}")
            # Fallback: split text into equal parts
            return self._fallback_split(text, min_chapters)
    
    def _fallback_split(self, text: str, num_parts: int) -> Dict[str, str]:
        """Fallback method: split text into equal parts.
        
        Args:
            text: Text to split.
            num_parts: Number of parts to split into.
            
        Returns:
            Dictionary mapping part numbers to text.
        """
        words = text.split()
        words_per_part = len(words) // num_parts
        
        chapters = {}
        for i in range(num_parts):
            start = i * words_per_part
            end = (i + 1) * words_per_part if i < num_parts - 1 else len(words)
            
            part_words = words[start:end]
            part_text = " ".join(part_words)
            
            chapters[f"Part {i + 1}"] = part_text
        
        return chapters
    
    def preprocess_book(self, file_path: str) -> Dict:
        """Complete preprocessing pipeline for a single book.
        
        Args:
            file_path: Path to the book file.
            
        Returns:
            Fully preprocessed book data.
        """
        logger.info(f"Preprocessing book: {file_path}")
        
        # Load and clean
        book_data = self.load_and_clean_book(file_path)
        
        # Split into chapters
        chapters = self.split_into_chapters(book_data["cleaned_text"])
        
        if len(chapters) < 4:
            logger.warning(f"Book has only {len(chapters)} chapters, quality may be low")
        
        return {
            "title": book_data["title"],
            "raw_text": book_data["raw_text"],
            "chapters": chapters
        }