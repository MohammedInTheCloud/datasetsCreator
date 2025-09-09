"""
Synthetic prompt generator for creating user prompts in the exact style of the LongPage dataset.
This generator prioritizes narrative specificity, character arcs, and unique world elements to mimic professional client briefs.
"""
import random
import logging
from typing import Dict, List, Any
import json

from agents.llm_agent import llm_agent

logger = logging.getLogger(__name__)


class SyntheticPromptGenerator:
    """Generates sophisticated, LongPage-style synthetic user prompts based on comprehensive book metadata."""

    def __init__(self):
        """Initialize the prompt generator with configuration for fallback methods."""
        # Define components for fallback generation (used if LLM fails)
        self.tones = {
            "neutral": {"weight": 0.4, "templates": [
                "Write a novel",
                "I need a book",
                "Create a story",
                "Generate a narrative"
            ]},
            "casual": {"weight": 0.25, "templates": [
                "Hey, can you write a novel",
                "I'd like to read a book",
                "Could you craft a story",
                "Make me a narrative"
            ]},
            "formal": {"weight": 0.25, "templates": [
                "Please compose a novel",
                "I require a literary work",
                "Kindly create a narrative",
                "I would appreciate a book"
            ]},
            "urgent": {"weight": 0.1, "templates": [
                "Quick! I need a novel",
                "Emergency! Write a book now",
                "Immediately create a story",
                "Urgent: Generate a narrative"
            ]}
        }

        self.length_indicators = {
            "short": {"words": 20000, "weight": 0.15},
            "medium": {"words": 50000, "weight": 0.5},
            "long": {"words": 100000, "weight": 0.35}
        }

    def generate_prompt(self, book_data: Dict[str, Any]) -> str:
        """Generate a sophisticated, LongPage-style synthetic user prompt based on book data.

        This method attempts to generate a rich, narrative prompt using an LLM. If that fails, it falls back to a simpler template-based method.

        Args:
            book_data: Dictionary containing comprehensive book metadata (title, arcs, rules, characters, style, chapters).

        Returns:
            Generated synthetic prompt string, ideally in LongPage format.
        """
        logger.debug("Generating sophisticated LongPage-style synthetic prompt")

        try:
            # Primary method: Generate a sophisticated prompt using the LLM
            sophisticated_prompt = self._generate_llm_prompt(book_data)
            if sophisticated_prompt and len(sophisticated_prompt.strip()) > 100:  # Sanity check for meaningful content
                logger.debug("Successfully generated sophisticated LongPage-style prompt.")
                return sophisticated_prompt.strip()
            else:
                logger.warning("LLM prompt generation returned an empty or very short result. Falling back to template method.")
        except Exception as e:
            logger.error(f"Failed to generate sophisticated LLM prompt: {e}. Falling back to template method.")

        # Fallback: Use the original, simpler method
        return self._generate_fallback_prompt(book_data)

    def _generate_llm_prompt(self, book_data: Dict[str, Any]) -> str:
        """Generate a detailed, natural-sounding user prompt in the LongPage style using an LLM.

        This is the core method that transforms metadata into a narrative client brief.
        """
        # Extract and prepare book information for the LLM context
        book_title = book_data.get("book_title", "Untitled Manuscript")
        book_highlight = book_data.get("book_highlight", "A compelling narrative with rich characters and a unique setting.")
        book_archetype = book_data.get("book_archetype", "General Fiction")

        # Get story arcs
        story_arcs_raw = book_data.get("story_arcs", [])
        story_arcs = []
        for arc in story_arcs_raw[:3]:  # Limit to first 3 for context
            if isinstance(arc, str):
                story_arcs.append(arc)
            elif isinstance(arc, dict) and 'description' in arc:
                # Handle case where arc is a dict with a description field
                story_arcs.extend(arc['description'][:2])  # Take first 2 bullets of description
            elif isinstance(arc, list):
                story_arcs.extend(arc[:2])  # If it's a list of strings, take first 2

        # Get world rules and select a "quirky" one to highlight
        world_rules = book_data.get("world_rules", [])
        quirky_rule = None
        if world_rules:
            # Simple heuristic: pick the longest rule as it's often the most detailed/unique
            quirky_rule = max(world_rules[:5], key=len, default=None)  # Consider first 5, pick longest

        # Get main characters and attempt to infer/formulate their arcs
        main_characters_data = book_data.get("book_characters", {}).get("main_characters", [])
        character_arcs = []
        for char in main_characters_data[:2]:  # Focus on 1-2 main characters
            name = char.get('name', 'Unnamed Protagonist')
            descriptions = char.get('description', [])
            # Create a simple arc string from the first few descriptions
            if descriptions:
                start_state = descriptions[0] if len(descriptions) > 0 else "a person with a secret"
                key_challenge = descriptions[1] if len(descriptions) > 1 else "facing a great trial"
                end_state = descriptions[-1] if len(descriptions) > 2 else "forever changed"
                arc_str = f"{name}: Begins as {start_state} -> Challenged by {key_challenge} -> Ends as {end_state}"
                character_arcs.append(arc_str)
            else:
                character_arcs.append(f"{name}: A central figure whose journey drives the narrative.")

        # Get writing style notes
        writing_style_notes = book_data.get("writing_style", [])

        # Estimate total word count from chapters
        total_word_count = self._estimate_total_words(book_data)
        # Estimate chapter count
        chapter_count = len(book_data.get("book_chapters", {}))
        chapter_range = f"{max(1, chapter_count - 5)}-{chapter_count + 5}" if chapter_count > 0 else "40-60"

        # Build the comprehensive context prompt for the LLM
        context_prompt = f"""
You are a literary agent or a demanding client commissioning a professional novelist. Your task is to generate a detailed, natural-sounding user prompt for a complete novel manuscript. The prompt must sound like it was written by a human with a fully formed story in mind, not an AI or a vague request.

Use the following book metadata to craft the prompt:

- **Core Concept/Highlight**: {book_highlight}
- **Working Title**: {book_title}
- **Genre/Archetype**: {book_archetype}
- **Key Story Arcs (Use these to build a 3-act plot structure)**:
{json.dumps(story_arcs[:3], indent=2)}
- **World Rules**: 
{json.dumps(world_rules[:3], indent=2)}
  - **CRITICAL**: You MUST include ONE specific, quirky, or pivotal world element as a "key element" or "non-negotiable feature" (e.g., "a 'Cyber-Bazaar'"). If provided, use this: "{quirky_rule}".
- **Character Arcs (Show transformation through action, not just traits)**:
{json.dumps(character_arcs, indent=2)}
- **Writing Style Notes**: {json.dumps(writing_style_notes[:5], indent=2)}

The generated prompt MUST adhere to the following structure and content guidelines:

1.  **Opening Request**: Start with a direct, professional request (e.g., "I need a complete novel manuscript for...", "Compose a story about...").
2.  **Mini-Synopsis (The Heart of the Prompt)**: Provide a 4-6 sentence narrative synopsis structured as a clear 3-act plot:
    - **ACT 1 (Setup & Inciting Incident)**: Introduce the protagonist and the event that disrupts their world.
    - **ACT 2 (Rising Action & Complication)**: Describe the journey, key challenges, and how relationships evolve. Include the "quirky" world element here.
    - **ACT 3 (Climax & Resolution)**: State the final confrontation and the protagonist's transformed state.
3.  **Tonal Guidance**: Specify the desired voice and tone (e.g., "wry, satirical, and formal" or "lyrical and melancholic") based on the writing style notes.
4.  **Structural Requirements**: State the target length as a precise range (e.g., "{total_word_count - 10000:,}-{total_word_count + 10000:,} words") and chapter count (e.g., "divided into {chapter_range} chapters").
5.  **Closing Instruction**: End with a firm, specific directive like "No preamble—start directly with the manuscript." or "Begin with Chapter One."

Output ONLY the final prompt, enclosed within <longPage> and </longPage> tags. Do not include any additional commentary, explanations, or labels.

Example of desired output format:
<longPage>
I need a complete novel manuscript for a new project. The story is a social satire set in the Victorian era, following the redemption of a selfish young man, Martin Chuzzlewit. After being disinherited by his wealthy grandfather, he is forced into an apprenticeship with the hypocritical Mr. Pecksniff. When Pecksniff's fraud is exposed, Martin, now penniless, travels with his unwaveringly optimistic companion Mark Tapley to America to make his fortune. There, they are swindled into buying land in a deadly swamp settlement called Eden, a humbling experience that transforms Martin's character. Upon his return to England, he must confront not only Pecksniff but also the brutal villainy of his cousin, Jonas Chuzzlewit. A key element is an unexplained pocket of futuristic technology, a 'Cyber-Bazaar,' that exists within the otherwise realistic 19th-century world. The manuscript should be in the 170k-190k word range, divided into 50-58 chapters. Please use a wry, satirical, and formal voice suitable for an advanced reading level. No preamble—start directly with the manuscript.
</longPage>
        """

        # Query the LLM to generate the prompt
        llm_response = llm_agent.query(context_prompt, max_tokens=8192, temperature=0.7)

        # Extract the content between <longPage> tags
        start_tag = "<longPage>"
        end_tag = "</longPage>"
        start_idx = llm_response.find(start_tag)
        end_idx = llm_response.find(end_tag)

        if start_idx != -1 and end_idx != -1:
            prompt_content = llm_response[start_idx + len(start_tag):end_idx].strip()
            return prompt_content
        else:
            # If tags are not found, return the raw response (it might still be usable)
            logger.warning("LLM response did not contain <longPage> tags. Returning raw output.")
            return llm_response.strip()

    def _estimate_total_words(self, book_data: Dict[str, Any]) -> int:
        """Estimate the total word count of the book by summing chapter word counts or estimating from text.

        Args:
            book_data: The book's metadata dictionary.

        Returns:
            Estimated total word count. Defaults to 80,000 if no data is available.
        """
        chapters = book_data.get("book_chapters", {})
        total_words = 0

        for chapter_key, chapter_data in chapters.items():
            if isinstance(chapter_data, dict):
                # If 'word_count' is directly available, use it
                if 'word_count' in chapter_data and isinstance(chapter_data['word_count'], int):
                    total_words += chapter_data['word_count']
                else:
                    # Fallback: estimate from the chapter text
                    chapter_text = chapter_data.get('chapter', '')
                    if isinstance(chapter_text, str):
                        total_words += len(chapter_text.split())

        # Set a reasonable default if no chapters or word counts are found
        return max(total_words, 80000)

    def _generate_fallback_prompt(self, book_data: Dict[str, Any]) -> str:
        """Generate a synthetic user prompt using a simple template method as a fallback.

        This is a simplified version focusing on core elements for reliability.

        Args:
            book_data: Dictionary containing book metadata.

        Returns:
            Generated synthetic prompt string.
        """
        logger.debug("Generating fallback prompt")

        book_title = book_data.get("book_title", "Unknown Title")
        book_highlight = book_data.get("book_highlight", "")
        book_archetype = book_data.get("book_archetype", "")
        world_rules = book_data.get("world_rules", [])
        story_arcs = book_data.get("story_arcs", [])

        # Choose a random tone
        tone = self._weighted_choice(
            list(self.tones.keys()),
            [t["weight"] for t in self.tones.values()]
        )

        # Choose a length
        length = self._weighted_choice(
            list(self.length_indicators.keys()),
            [l["weight"] for l in self.length_indicators.values()]
        )
        word_count = self.length_indicators[length]["words"]

        # Build a simple, direct prompt
        template = random.choice(self.tones[tone]["templates"])
        prompt = f"{template} titled '{book_title}'."

        if book_highlight:
            prompt += f" {book_highlight}"
        if book_archetype:
            prompt += f" Genre: {book_archetype}."

        # Add a random world rule if available
        if world_rules and random.random() < 0.5:
            rule = random.choice(world_rules)
            prompt += f" Key element: {rule}"

        # Add length
        prompt += f" Target length: approximately {word_count:,} words."

        # Add a standard closing
        prompt += " No preamble—start directly with the manuscript."

        return prompt

    def _weighted_choice(self, options: List[str], weights: List[float]) -> str:
        """Make a weighted random choice.

        Args:
            options: List of options to choose from.
            weights: List of weights for each option.

        Returns:
            Chosen option.
        """
        total = sum(weights)
        r = random.uniform(0, total)
        upto = 0
        for option, weight in zip(options, weights):
            if upto + weight >= r:
                return option
            upto += weight
        return options[-1]

    def generate_book_highlight(self, book_data: Dict[str, Any]) -> str:
        """Generate a concise, compelling book highlight (15-25 words).

        Args:
            book_data: Dictionary containing book metadata.

        Returns:
            Generated book highlight string.
        """
        logger.debug("Generating book highlight")

        book_title = book_data.get("book_title", "Unknown Title")
        world_rules = book_data.get("world_rules", [])
        story_arcs = book_data.get("story_arcs", [])
        character_archetypes = book_data.get("character_archetypes", {})

        # Build context for the LLM
        context_parts = []

        if world_rules:
            context_parts.append(f"World rules: {', '.join(world_rules[:2])}")

        if story_arcs:
            # Flatten story arcs if they are lists of strings
            flat_arcs = []
            for arc in story_arcs[:2]:
                if isinstance(arc, list):
                    flat_arcs.extend(arc[:1])  # Take first bullet from each arc
                else:
                    flat_arcs.append(arc)
            context_parts.append(f"Main story arcs: {', '.join(flat_arcs)}")

        if character_archetypes:
            chars = list(character_archetypes.items())[:3]
            char_desc = ", ".join([f"{name} ({', '.join(archetype[:1]) if isinstance(archetype, list) else archetype})" for name, archetype in chars])
            context_parts.append(f"Key characters: {char_desc}")

        context = "\n".join(context_parts)

        prompt = f"""
Create a concise, compelling highlight (15-25 words) for a book titled "{book_title}" that captures its core essence, mood, and hook.

Context:
{context}

Highlight:
        """

        try:
            highlight = llm_agent.query(prompt, max_tokens=8192, temperature=0.4)
            return highlight.strip()
        except Exception as e:
            logger.error(f"Failed to generate book highlight: {e}")
            return f"A gripping {book_archetype or 'narrative'} of transformation and discovery."


# Singleton instance for easy import and use
synthetic_prompt_generator = SyntheticPromptGenerator()