#!/usr/bin/env python3
"""
Process the extracted JSON file and save prompt, thinking, and book components.
"""

import json
from exampel_compose import compose
import os

def load_and_process_json(json_path):
    """Load the JSON file and process it with compose function."""
    
    # Load the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Transform the data to match the expected format for compose()
    transformed_data = {
        "use_prompt": data.get("use_prompt", ""),
        "book_highlight": data.get("book_highlight", ""),
        "book_title": data.get("book_title", ""),
        "book_tags": data.get("book_tags", []),
        "book_archetype": data.get("book_archetype", ""),
        "world_rules": data.get("world_rules", []),
        "story_arcs": [],
        "character_archetypes": {},
        "writing_style": data.get("writing_style", []),
        "book_characters": {
            "main_characters": [],
            "side_characters": []
        },
        "book_chapters": {}
    }
    
    # Transform story arcs
    if isinstance(data.get("story_arcs"), list):
        for i, arc in enumerate(data["story_arcs"]):
            transformed_data["story_arcs"].append({
                "arc_name": f"Arc {i+1}",
                "arc_summary": [arc] if isinstance(arc, str) else arc
            })
    
    # Transform character archetypes
    if isinstance(data.get("character_archetypes"), dict):
        for char, archetype in data["character_archetypes"].items():
            if isinstance(archetype, str):
                transformed_data["character_archetypes"][char] = [f"Archetype: {archetype}"]
            else:
                transformed_data["character_archetypes"][char] = archetype if isinstance(archetype, list) else [str(archetype)]
    
    # Transform characters
    if isinstance(data.get("book_characters"), dict):
        for char_type in ["main_characters", "side_characters"]:
            if char_type in data["book_characters"]:
                for char_name in data["book_characters"][char_type]:
                    transformed_data["book_characters"][char_type].append({
                        "name": char_name,
                        "description": [f"Character from the book: {char_name}"]
                    })
    
    # Transform chapters
    if isinstance(data.get("book_chapters"), dict):
        for chapter_name, chapter_data in data["book_chapters"].items():
            if isinstance(chapter_data, dict):
                transformed_data["book_chapters"][chapter_name] = {
                    "chapter": chapter_data.get("chapter", ""),
                    "embedding_space": chapter_data.get("embedding_space", {
                        "action": 0, "dialog": 0, "pacing": 0, "exposition": 0,
                        "romantic": 0, "world_building": 0, "erotic": 0
                    }),
                    "chapter_summary": chapter_data.get("chapter_summary", []),
                    "scene_breakdown": chapter_data.get("scene_breakdown", [])
                }
            else:
                # If chapter_data is just a string
                transformed_data["book_chapters"][chapter_name] = {
                    "chapter": str(chapter_data),
                    "embedding_space": {
                        "action": 0, "dialog": 0, "pacing": 0, "exposition": 0,
                        "romantic": 0, "world_building": 0, "erotic": 0
                    },
                    "chapter_summary": ["Chapter content"],
                    "scene_breakdown": []
                }
    
    return transformed_data

def main():
    json_path = "longpage_dataset_20250908_152850.json"
    
    print("Loading and processing JSON file...")
    transformed_data = load_and_process_json(json_path)
    
    print("Applying compose function...")
    prompt, thinking_markdown, book_markdown = compose(transformed_data)
    
    # Create output directory
    output_dir = "extracted_components"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save components
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    
    with open(os.path.join(output_dir, f"{base_name}_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)
    
    with open(os.path.join(output_dir, f"{base_name}_thinking.md"), "w", encoding="utf-8") as f:
        f.write(thinking_markdown)
    
    with open(os.path.join(output_dir, f"{base_name}_book.md"), "w", encoding="utf-8") as f:
        f.write(book_markdown)
    
    print(f"\nComponents saved to {output_dir}/")
    print(f"- {base_name}_prompt.txt")
    print(f"- {base_name}_thinking.md")
    print(f"- {base_name}_book.md")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"- Prompt length: {len(prompt)} characters")
    print(f"- Thinking length: {len(thinking_markdown)} characters")
    print(f"- Book length: {len(book_markdown)} characters")

if __name__ == "__main__":
    main()
