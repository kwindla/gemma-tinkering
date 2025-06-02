#!/usr/bin/env python3
"""FastAPI server for Gemma model streaming inference."""

import asyncio
import queue
import threading
from typing import Optional

import mlx.core as mx
import mlx_lm
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import cmudict

# Optional: Set environment variables for offline mode
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class PromptRequest(BaseModel):
    messages: Optional[list] = None  # For multi-turn conversations
    prompt: Optional[str] = None  # For backward compatibility
    max_tokens: Optional[int] = 512
    use_syllable_filter: Optional[bool] = False
    syllable_count: Optional[int] = 2


app = FastAPI()

# Global model and tokenizer
model = None
tokenizer = None
cmu_dict = None


def load_model(model_name: str = "mlx-community/gemma-3-4b-it-8bit"):
    """Load the model and tokenizer."""
    global model, tokenizer, cmu_dict
    print(f"Loading model: {model_name}")
    model, tokenizer = mlx_lm.load(model_name)
    
    # Initialize CMU dictionary for syllable counting
    if cmu_dict is None:
        cmu_dict = cmudict.dict()
    
    print("Model loaded successfully")


def count_syllables(word: str) -> int:
    """Count syllables in a word using CMU dictionary."""
    word_lower = word.lower()
    if word_lower in cmu_dict:
        # CMU dict returns list of pronunciations, we use the first one
        # Each digit in pronunciation represents a stressed vowel (syllable)
        return len([ph for ph in cmu_dict[word_lower][0] if ph[-1].isdigit()])
    # Fallback: count vowels as rough approximation
    vowels = "aeiouAEIOU"
    return sum(1 for char in word if char in vowels)


def select_token(current_buffer, sorted_tokens, use_syllable_filter: bool, syllable_count: int):
    """Select a token based on syllable count or other criteria."""
    if not len(sorted_tokens):
        return None

    token_text = tokenizer.decode([int(sorted_tokens[0])])
    stripped = token_text.strip()

    # Select our prompted eol token or any whitespace tokens
    if stripped == "/" or len(stripped) == 0:
        return sorted_tokens[0]

    # Skip any token that is a single non-alphabetic character
    if len(stripped) == 1 and not stripped.isalpha():
        return select_token(current_buffer, sorted_tokens[1:], use_syllable_filter, syllable_count)

    # Apply syllable filter if requested
    if use_syllable_filter and count_syllables(stripped) == syllable_count:
        return sorted_tokens[0]
    elif not use_syllable_filter:
        # If no syllable filter, just return the top token
        return sorted_tokens[0]

    return select_token(current_buffer, sorted_tokens[1:], use_syllable_filter, syllable_count)


def run_inference(prompt_tokens: mx.array, max_tokens: int, token_queue: queue.Queue, 
                 use_syllable_filter: bool = False, syllable_count: int = 2):
    """Run inference synchronously and push tokens to queue."""
    try:
        context = prompt_tokens
        cache = None
        current_buffer = []

        for _ in range(max_tokens):
            # Try to use KV caching if supported
            try:
                if cache is None:
                    # First inference - process full prompt
                    result = model(context[None], cache=cache)
                else:
                    # Subsequent inferences - only process the new token
                    result = model(context[None, -1:], cache=cache)

                # Check if model returns cache
                if isinstance(result, tuple) and len(result) == 2:
                    logits, cache = result
                else:
                    # Model doesn't support caching
                    logits = result
                    cache = None
            except (TypeError, ValueError):
                # Fallback to non-cached inference if cache parameter not supported
                logits = model(context[None])
                cache = None

            # Get log probabilities
            log_probs = logits[0, -1, :] - mx.logsumexp(logits[0, -1, :])
            tops = mx.argsort(-log_probs)[:40]

            # Check if we're done
            if int(tops[0]) == tokenizer.eos_token_id:
                break

            # Select token based on criteria
            if use_syllable_filter:
                selected_token = select_token(current_buffer, tops, use_syllable_filter, syllable_count)
            else:
                selected_token = tops[0]

            if selected_token is None:
                break

            # Decode token
            token_text = tokenizer.decode([int(selected_token)])
            
            # Push to queue
            token_queue.put(token_text)

            # Update context
            context = mx.concatenate([context, mx.array([int(selected_token)])])

    except Exception as e:
        token_queue.put(e)
    finally:
        # Signal completion
        token_queue.put(None)


async def inference_generator(prompt_request: PromptRequest, 
                            use_syllable_filter: bool = False, 
                            syllable_count: int = 2):
    """Async generator that runs inference in background thread."""
    # Apply chat template and convert to tokens
    if prompt_request.messages:
        messages = prompt_request.messages
    elif prompt_request.prompt:
        messages = [{"role": "user", "content": prompt_request.prompt}]
    else:
        raise ValueError("Either messages or prompt must be provided")

    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_tokens = mx.array(prompt_tokens)

    token_queue = queue.Queue()

    # Start inference in background thread
    inference_thread = threading.Thread(
        target=run_inference,
        args=(prompt_tokens, prompt_request.max_tokens, token_queue, use_syllable_filter, syllable_count),
        daemon=True
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
                yield token

        except queue.Empty:
            # No token available yet, yield control
            await asyncio.sleep(0.001)


@app.post("/generate")
async def generate(prompt_request: PromptRequest):
    """Generate text from a prompt with streaming response."""
    if model is None:
        return {"error": "Model not loaded. Server may still be initializing."}

    async def stream_response():
        async for token in inference_generator(
            prompt_request, 
            prompt_request.use_syllable_filter,
            prompt_request.syllable_count
        ):
            yield token

    return StreamingResponse(stream_response(), media_type="text/plain")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Gemma FastAPI Server")
    parser.add_argument("--model", type=str, default="mlx-community/gemma-3-4b-it-8bit",
                        help="Model to load (default: mlx-community/gemma-3-4b-it-8bit)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind to (default: 8000)")
    
    args = parser.parse_args()

    # Load model before starting server
    load_model(args.model)

    # Start server with streaming configuration
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        # Disable buffering for better streaming
        access_log=False,
        loop="asyncio"
    )
