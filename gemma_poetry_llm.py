from typing import Optional
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.utils.tracing.service_decorators import traced_llm
from pipecat.frames.frames import LLMTextFrame

import mlx_lm
import mlx.core as mx
import asyncio
import threading
import queue

from loguru import logger


class GemmaPoetryLLMService(OpenAILLMService):
    def __init__(
        self,
        *,
        model: str = "mlx-community/gemma-3-12b-it-8bit",
        min_syllables: int = 1,
        params: Optional[OpenAILLMService.InputParams] = None,
        **kwargs,
    ):
        super().__init__(model=model, params=params, **kwargs)

        self.min_syllables = min_syllables
        logger.info(f"Loading model {model}")
        self._model, self._tokenizer = mlx_lm.load(model)
        logger.debug(f"Loaded model: {self._model}")
        logger.debug(f"Loaded tokenizer: {self._tokenizer}")

    def create_client(self, *args, **kwargs):
        logger.debug(f"Creating client with args: {args}, kwargs: {kwargs}")

    def _run_inference(self, prompt, max_tokens, token_queue):
        """Run inference synchronously and push tokens to queue"""
        current_prompt = prompt
        cache = None

        try:
            for i in range(max_tokens):
                # Try to use KV caching if supported
                try:
                    if cache is None:
                        # First inference - process full prompt
                        result = self._model(current_prompt[None], cache=cache)
                    else:
                        # Subsequent inferences - only process the new token
                        result = self._model(current_prompt[None, -1:], cache=cache)

                    # Check if model returns cache
                    if isinstance(result, tuple) and len(result) == 2:
                        logits, cache = result
                    else:
                        # Model doesn't support caching
                        logits = result
                        cache = None
                except (TypeError, ValueError):
                    # Fallback to non-cached inference if cache parameter not supported
                    logits = self._model(current_prompt[None])
                    cache = None

                # Get log probabilities
                log_probs = logits[0, -1, :] - mx.logsumexp(logits[0, -1, :])
                tops = mx.argsort(-log_probs)[:10]

                # Extract top token
                top_token_id = int(tops[0])

                # Check if we're done
                if top_token_id == self._tokenizer.eos_token_id:
                    break

                # Decode token
                token_text = self._tokenizer.decode([top_token_id])

                # Push to queue
                token_queue.put(token_text)

                # Update prompt
                current_prompt = mx.concatenate([current_prompt, mx.array([top_token_id])])

        except Exception as e:
            token_queue.put(e)
        finally:
            # Signal completion
            token_queue.put(None)

    async def _inference_loop(self, prompt, n):
        """Async generator that runs inference in background thread"""
        token_queue = queue.Queue()

        # Start inference in background thread
        inference_thread = threading.Thread(
            target=self._run_inference, args=(prompt, n, token_queue), daemon=True
        )
        inference_thread.start()

        # Yield tokens as they become available
        while True:
            try:
                # Check for tokens with a short timeout
                token = token_queue.get(timeout=0.01)

                if token is None:
                    # End of sequence
                    break
                elif isinstance(token, Exception):
                    raise token
                else:
                    logger.debug(f"!!! Token text: {token}")
                    yield token

            except queue.Empty:
                # No token available yet, yield control
                await asyncio.sleep(0.001)

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext):
        logger.debug(f"!!! {self}: Generating chat [{context.get_messages_for_logging()}]")

        messages = context.get_messages()
        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompt = mx.array(prompt)

        test = "Write a short poem about computer programming. The poem should have 4 lines. At the end of each line, add this punctuation ' /'\n\nOutput only the poem, nothing else."
        messages = [{"role": "user", "content": test}]
        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompt = mx.array(prompt)

        await self.start_ttfb_metrics()

        try:
            async for token_text in self._inference_loop(prompt, 128):
                await self.push_frame(LLMTextFrame(token_text))

        except Exception as e:
            logger.exception(f"Error in LLM processing: {str(e)}")
            raise

        await self.stop_ttfb_metrics()

    def __del__(self):
        """Cleanup"""
        pass
