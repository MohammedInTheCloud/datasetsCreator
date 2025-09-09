"""
Fixed version of llm_agent.py with proper think tag handling.
"""
import time
import logging
import re
from typing import Optional
from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Optional: Also force logging handler to use UTF-8
for handler in logging.getLogger().handlers:
    if hasattr(handler.stream, 'reconfigure'):
        handler.stream.reconfigure(encoding='utf-8')
        
# Configure logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
logger = logging.getLogger(__name__)

def strip_think_content(response: str) -> str:
    """
    Strip thinking content from model responses.
    Handles incomplete <think> blocks and logs warnings.
    """
    if not response:
        return response

    start_tag = "<think>"
    end_tag = "</think>"
    
    start_idx = response.find(start_tag)
    if start_idx == -1:
        return response  # No think block

    end_idx = response.find(end_tag, start_idx + len(start_tag))

    if end_idx == -1:
        # Incomplete think block — log warning and return content BEFORE think
        snippet = response[start_idx:start_idx + 50].replace('\n', '\\n')
        logging.warning(
            "(strip_think) ⚠️  INCOMPLETE <think> BLOCK DETECTED!\n"
            f"Start found at index {start_idx}, but no closing '{end_tag}' found.\n"
            f"Raw snippet: \"{snippet}\"..."
        )
        # Return everything BEFORE the incomplete think block
        return response[:start_idx].rstrip()

    # Extract content after </think>
    clean_content = response[end_idx + len(end_tag):].lstrip()
    return clean_content

class LLMAgent:
    """Core LLM agent for handling all API calls to Groq."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the LLM agent."""
        # Get configuration from environment or parameters
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "llama3-70b-8192")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        
        if not self.api_key:
            raise ValueError("API key must be provided either directly or via LLM_API_KEY environment variable")
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.rate_limit_delay = float(os.getenv("RATE_LIMIT_DELAY", "1"))
        
    def query(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.2,
        retries: Optional[int] = None
    ) -> str:
        """Query the LLM with exponential backoff for rate limits."""
        model = model or self.model
        retries = retries or self.max_retries
        
        for attempt in range(retries):
            try:
                logger.debug(f"LLM query attempt {attempt + 1}/{retries}")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                raw_content = response.choices[0].message.content.strip()
                # Strip thinking content if present
                clean_content = strip_think_content(raw_content)
                return clean_content
                
            except RateLimitError as e:
                if attempt < retries - 1:
                    wait_time = (2 ** attempt) + self.rate_limit_delay
                    logger.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rate limit error after {retries} attempts: {e}")
                    raise
                    
            except APIError as e:
                logger.error(f"API error: {e}")
                raise
                
            except Exception as e:
                logger.error(f"Unexpected error in LLM query: {e}")
                if attempt < retries - 1:
                    wait_time = (2 ** attempt) + self.rate_limit_delay
                    logger.warning(f"Unexpected error. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise APIError(f"Failed after {retries} attempts: {e}")
        
        return ""

    def query_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.2,
        retries: Optional[int] = None
    ) -> dict:
        """Query the LLM and parse the response as JSON."""
        import json
        
        # Add JSON formatting instruction to prompt
        json_prompt = prompt + "\n\nPlease respond with valid JSON only."
        
        response = self.query(
            prompt=json_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=retries
        )
        
        try:
            # Try to parse as JSON directly
            return json.loads(response)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the response
            # Look for JSON between ```json and ``` markers
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Look for JSON between { and } braces
            brace_match = re.search(r'\{.*\}', response, re.DOTALL)
            if brace_match:
                try:
                    return json.loads(brace_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, raise an error
            raise ValueError(f"Could not parse JSON from LLM response: {response[:500]}...")

# Singleton instance for easy import
llm_agent = LLMAgent()