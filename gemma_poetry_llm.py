from typing import Optional
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.utils.tracing.service_decorators import traced_llm
from pipecat.frames.frames import LLMTextFrame

import aiohttp

from loguru import logger


class GemmaPoetryLLMService(OpenAILLMService):
    def __init__(
        self,
        *,
        server_url: str = "http://localhost:8000",
        model: str = "mlx-community/gemma-3-12b-it-8bit",
        min_syllables: int = 1,
        use_syllable_filter: bool = False,
        params: Optional[OpenAILLMService.InputParams] = None,
        **kwargs,
    ):
        super().__init__(model=model, params=params, **kwargs)

        self.server_url = server_url.rstrip("/")
        self.min_syllables = min_syllables
        self.use_syllable_filter = use_syllable_filter
        logger.info(f"Using Gemma server at {self.server_url}")

        # Note: We no longer load the model here - the server handles that

    def create_client(self, *args, **kwargs):
        logger.debug(f"Creating client with args: {args}, kwargs: {kwargs}")

    async def _inference_loop(self, messages: list, max_tokens: int):
        """Stream tokens from the HTTP server"""
        url = f"{self.server_url}/generate"

        # Prepare the request payload
        payload = {
            "messages": messages,  # Send full messages array
            "max_tokens": max_tokens,
            "use_syllable_filter": self.use_syllable_filter,
            "syllable_count": self.min_syllables,
        }

        # Make streaming HTTP request with proper configuration
        timeout = aiohttp.ClientTimeout(total=None)  # No timeout for streaming
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    chunked=True,  # Enable chunked transfer encoding
                ) as response:
                    response.raise_for_status()

                    # Stream the response chunk by chunk
                    async for data in response.content.iter_any():
                        if data:
                            token_text = data.decode("utf-8", errors="ignore")
                            if token_text:  # Only yield non-empty tokens
                                logger.debug(f"!!! Token text: {token_text!r}")
                                yield token_text

            except aiohttp.ClientError as e:
                logger.exception(f"HTTP request failed: {str(e)}")
                raise

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext):
        logger.debug(f"!!! {self}: Generating chat [{context.get_messages_for_logging()}]")

        messages = context.get_messages()

        await self.start_ttfb_metrics()

        try:
            async for token_text in self._inference_loop(messages, 2048):
                await self.push_frame(LLMTextFrame(token_text))

        except Exception as e:
            logger.exception(f"Error in LLM processing: {str(e)}")
            raise

        await self.stop_ttfb_metrics()

    def __del__(self):
        """Cleanup"""
        pass
