"""
Async version of the LLM agent with parallel processing capabilities.
Extends the existing llm_agent.py with async methods.
"""
import asyncio
import logging
import time
from typing import Optional, Dict, List
import aiohttp
from openai import OpenAI, RateLimitError, APIError
import os
import sys

# Import the existing agent for shared functionality
from agents.llm_agent import llm_agent, strip_think_content

logger = logging.getLogger(__name__)


class AsyncLLMAgent:
    """Async LLM agent for handling concurrent API calls."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the async LLM agent."""
        # Use the same configuration as the sync agent
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "llama3-70b-8192")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")

        if not self.api_key:
            raise ValueError("API key must be provided either directly or via LLM_API_KEY environment variable")

        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.rate_limit_delay = float(os.getenv("RATE_LIMIT_DELAY", "1"))

        # For async operations, we'll use aiohttp directly
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def query_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.2,
        retries: Optional[int] = None,
        session: Optional[aiohttp.ClientSession] = None
    ) -> str:
        """Async query the LLM with exponential backoff for rate limits."""
        model = model or self.model
        retries = retries or self.max_retries

        # Use provided session or create temporary one
        close_session = False
        if session is None:
            session = aiohttp.ClientSession(headers=self.headers)
            close_session = True

        try:
            for attempt in range(retries):
                try:
                    logger.debug(f"Async LLM query attempt {attempt + 1}/{retries}")

                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }

                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        json=payload
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            raw_content = data["choices"][0]["message"]["content"].strip()
                            # Strip thinking content if present
                            clean_content = strip_think_content(raw_content)
                            return clean_content
                        elif response.status == 429:  # Rate limit
                            if attempt < retries - 1:
                                wait_time = (2 ** attempt) + self.rate_limit_delay
                                logger.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                logger.error(f"Rate limit error after {retries} attempts")
                                raise RateLimitError("Rate limit exceeded")
                        else:
                            error_text = await response.text()
                            raise APIError(f"API error: {response.status} - {error_text}")

                except RateLimitError as e:
                    if attempt < retries - 1:
                        wait_time = (2 ** attempt) + self.rate_limit_delay
                        logger.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Rate limit error after {retries} attempts: {e}")
                        raise

                except APIError as e:
                    logger.error(f"API error: {e}")
                    raise

                except Exception as e:
                    logger.error(f"Unexpected error in async LLM query: {e}")
                    if attempt < retries - 1:
                        wait_time = (2 ** attempt) + self.rate_limit_delay
                        logger.warning(f"Unexpected error. Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise APIError(f"Failed after {retries} attempts: {e}")

            return ""

        finally:
            if close_session:
                await session.close()

    async def query_json_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.2,
        retries: Optional[int] = None,
        session: Optional[aiohttp.ClientSession] = None
    ) -> dict:
        """Async query the LLM and parse the response as JSON."""
        import json

        # Add JSON formatting instruction to prompt
        json_prompt = prompt + "\n\nPlease respond with valid JSON only."

        response = await self.query_async(
            prompt=json_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=retries,
            session=session
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

    async def batch_query_async(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.2,
        concurrent_limit: int = 10
    ) -> List[str]:
        """
        Make multiple LLM queries concurrently.

        Args:
            prompts: List of prompts to query
            model: Model to use
            max_tokens: Maximum tokens per response
            temperature: Temperature for responses
            concurrent_limit: Maximum concurrent requests

        Returns:
            List of responses in the same order as prompts
        """
        logger.debug(f"Making {len(prompts)} LLM queries with concurrency limit {concurrent_limit}")

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_limit)

        # Create shared session
        async with aiohttp.ClientSession(headers=self.headers) as session:

            async def query_with_semaphore(prompt: str, index: int) -> tuple[int, str]:
                async with semaphore:
                    try:
                        response = await self.query_async(
                            prompt, model, max_tokens, temperature, session=session
                        )
                        return index, response
                    except Exception as e:
                        logger.error(f"Query {index} failed: {e}")
                        return index, f"Error: {str(e)}"

            # Create tasks for all prompts
            tasks = [query_with_semaphore(prompt, i) for i, prompt in enumerate(prompts)]

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)

            # Sort results by index and extract responses
            sorted_results = sorted(results, key=lambda x: x[0])
            responses = [result[1] for result in sorted_results]

            return responses


# Singleton instance for easy import
async_llm_agent = AsyncLLMAgent()


# Hybrid agent that can fall back to sync operations
class HybridLLMAgent:
    """
    Hybrid LLM agent that can use both sync and async operations.
    Falls back to sync agent when async is not available.
    """

    def __init__(self):
        """Initialize hybrid agent."""
        self.async_agent = async_llm_agent
        self.sync_agent = llm_agent

    async def query_async(self, *args, **kwargs) -> str:
        """Async query method."""
        return await self.async_agent.query_async(*args, **kwargs)

    async def query_json_async(self, *args, **kwargs) -> dict:
        """Async JSON query method."""
        return await self.async_agent.query_json_async(*args, **kwargs)

    async def batch_query_async(self, *args, **kwargs) -> List[str]:
        """Batch async query method."""
        return await self.async_agent.batch_query_async(*args, **kwargs)

    def query(self, *args, **kwargs) -> str:
        """Sync query method (fallback)."""
        return self.sync_agent.query(*args, **kwargs)

    def query_json(self, *args, **kwargs) -> dict:
        """Sync JSON query method (fallback)."""
        return self.sync_agent.query_json(*args, **kwargs)


# Global hybrid instance
hybrid_llm_agent = HybridLLMAgent()