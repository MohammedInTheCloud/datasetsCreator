"""
Improved think tag stripping implementation.
"""
import re
import logging

logger = logging.getLogger(__name__)

def strip_think_content_improved(response: str) -> str:
    """
    Strip thinking content from model responses.
    
    This improved version properly handles:
    - Multiple think sections
    - Unclosed think tags
    - Various tag formats
    
    Args:
        response: The raw response from the model
        
    Returns:
        The response with thinking content removed
    """
    if not response:
        return response
    
    # First, try to remove properly formatted think sections
    pattern = r'</think>.*?</think>'
    clean_response = re.sub(pattern, '', response, flags=re.DOTALL)
    
    # Handle unclosed think tags
    if '</think>' in clean_response:
        # Find all unclosed think tags and remove everything after the last one
        parts = clean_response.split('</think>')
        
        # Keep everything before the first unclosed tag
        if len(parts) > 1:
            # Check if any part has a closing tag
            has_closed_tag = any('</think>' in part for part in parts[1:])
            
            if not has_closed_tag:
                # No closing tags found, keep only content before first think tag
                clean_response = parts[0]
                logger.warning("Removed content after unclosed think tag")
            else:
                # Some sections are properly closed, reconstruct carefully
                result = [parts[0]]
                i = 1
                while i < len(parts):
                    if i + 1 < len(parts) and '</think>' in parts[i + 1]:
                        # This is a properly closed section, skip both parts
                        i += 2
                    else:
                        # This is unclosed, skip it
                        i += 1
                clean_response = ''.join(result)
    
    # Final cleanup
    clean_response = clean_response.strip()
    
    # Log if we made changes
    if clean_response != response:
        logger.info(f"Stripped think content from response (reduced from {len(response)} to {len(clean_response)} chars)")
    
    return clean_response