"""
Utility function to strip think tags from model responses.
"""
import re
import logging

logger = logging.getLogger(__name__)

def strip_think_content(response: str) -> str:
    """
    Strip thinking content from model responses.
    
    Some models have a "think" feature where they output their reasoning process
    within </think>...</think> tags. This function removes the thinking content
    and keeps only the final response after the closing  tag.
    
    Args:
        response: The raw response from the model
        
    Returns:
        The response with thinking content removed
    """
    if not response:
        return response
    
    # Pattern to match think tags and content between them
    # Using non-greedy match to handle multiple think sections
    pattern = r'</think>.*?</think>'
    
    # Replace all think sections with empty string
    clean_response = re.sub(pattern, '', response, flags=re.DOTALL)
    
    # Also handle case where there might be only opening tag
    if '</think>' in clean_response and '</think>' not in clean_response:
        logger.warning("Found opening  tag but no closing  tag")
    
    return clean_response.strip()