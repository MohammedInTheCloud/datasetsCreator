"""
Composer function for formatting the final output.
"""
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def compose(data: Dict[str, Any]) -> tuple[str, str, str]:
    """
    Compose the final output into three parts: prompt, thinking, and book.
    
    Args:
        data: The complete book data dictionary
        
    Returns:
        tuple: (user_prompt, thinking_markdown, book_markdown)
    """
    logger.info("Composing final output")
    
    # Part 1: User Prompt
    user_prompt = data.get("use_prompt", "")
    
    # Part 2: Thinking Markdown
    thinking_markdown = _generate_thinking_markdown(data)
    
    # Part 3: Book Markdown
    book_markdown = _generate_book_markdown(data)
    
    return user_prompt, thinking_markdown, book_markdown


def _generate_thinking_markdown(data: Dict[str, Any]) -> str:
    """Generate the thinking/markdown part of the output."""
    
    lines = []
    
    # Book metadata
    lines.append(f"## 📖 {data.get('book_title', 'Unknown Title')}")
    lines.append("")
    
    if data.get("book_highlight"):
        lines.append(f"**Highlight:** {data['book_highlight']}")
        lines.append("")
    
    if data.get("book_archetype"):
        lines.append(f"**Archetype:** {data['book_archetype']}")
        lines.append("")
    
    if data.get("book_tags"):
        lines.append(f"**Tags:** {', '.join(data['book_tags'])}")
        lines.append("")
    
    # World Rules
    if data.get("world_rules"):
        lines.append("### 🌍 World Rules")
        lines.append("")
        for rule in data["world_rules"]:
            lines.append(f"- {rule}")
        lines.append("")
    
    # Story Arcs
    if data.get("story_arcs"):
        lines.append("### 📈 Story Arcs")
        lines.append("")
        for arc in data["story_arcs"]:
            lines.append(f"- {arc}")
        lines.append("")
    
    # Characters
    if data.get("book_characters"):
        chars = data["book_characters"]
        lines.append("### 👥 Characters")
        lines.append("")
        
        if chars.get("main_characters"):
            lines.append("**Main Characters:**")
            for char in chars["main_characters"]:
                archetype = data.get("character_archetypes", {}).get(char, "")
                if archetype:
                    lines.append(f"- {char} ({archetype})")
                else:
                    lines.append(f"- {char}")
            lines.append("")
        
        if chars.get("side_characters"):
            lines.append("**Side Characters:**")
            for char in chars["side_characters"]:
                archetype = data.get("character_archetypes", {}).get(char, "")
                if archetype:
                    lines.append(f"- {char} ({archetype})")
                else:
                    lines.append(f"- {char}")
            lines.append("")
    
    # Writing Style
    if data.get("writing_style"):
        lines.append("### ✍️ Writing Style")
        lines.append("")
        for style in data["writing_style"]:
            lines.append(f"- {style}")
        lines.append("")
    
    # Chapters overview
    if data.get("book_chapters"):
        lines.append("### 📚 Chapter Structure")
        lines.append("")
        
        for chapter_title, chapter_data in data["book_chapters"].items():
            lines.append(f"#### {chapter_title}")
            lines.append("")
            
            # Chapter summary
            if chapter_data.get("chapter_summary"):
                lines.append("**Summary:**")
                for point in chapter_data["chapter_summary"]:
                    lines.append(f"- {point}")
                lines.append("")
            
            # Embedding space
            if chapter_data.get("embedding_space"):
                embed = chapter_data["embedding_space"]
                lines.append("**Focus:**")
                focus_items = []
                for dim, value in embed.items():
                    if value > 30:  # Only show significant dimensions
                        focus_items.append(f"{dim.replace('_', ' ').title()}: {value}%")
                if focus_items:
                    lines.append(f"- {', '.join(focus_items)}")
                lines.append("")
            
            # Scene count
            scene_count = len(chapter_data.get("scene_breakdown", []))
            lines.append(f"**Scenes:** {scene_count}")
            lines.append("")
    
    return "\n".join(lines)


def _generate_book_markdown(data: Dict[str, Any]) -> str:
    """Generate the book/markdown part of the output."""
    
    lines = []
    
    # Title
    lines.append(f"# {data.get('book_title', 'Unknown Title')}")
    lines.append("")
    
    # Generate the actual book content
    if data.get("book_chapters"):
        for chapter_title, chapter_data in data["book_chapters"].items():
            lines.append(f"## {chapter_title}")
            lines.append("")
            
            # Add the chapter text
            chapter_text = chapter_data.get("chapter", "")
            if chapter_text:
                lines.append(chapter_text)
                lines.append("")
    
    return "\n".join(lines)