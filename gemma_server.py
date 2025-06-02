#!/usr/bin/env python3
"""FastAPI server for Gemma model streaming inference with KV caching."""

import asyncio
from typing import Optional

import mlx.core as mx
import mlx_lm
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import cmudict


class PromptRequest(BaseModel):
    prompt: str
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
        return len([ph for ph in cmu_dict[word_lower][0] if ph[-1].isdigit()])
    # Fallback: count vowels as rough approximation
    vowels = "aeiouAEIOU"
    return sum(1 for char in word if char in vowels)


def syllable_logits_processor(tokens: mx.array, logits: mx.array, syllable_count: int) -> mx.array:
    """Logits processor that filters tokens by syllable count."""
    # Get top candidates
    log_probs = logits - mx.logsumexp(logits)
    top_indices = mx.argsort(-log_probs)[:100]  # Consider top 100 tokens

    # Filter by syllable count
    filtered_indices = []
    for idx in top_indices:
        token_text = tokenizer.decode([int(idx)]).strip()

        # Always allow special tokens and whitespace
        if token_text == "/" or len(token_text) == 0:
            filtered_indices.append(idx)
            continue

        # Skip single non-alphabetic characters
        if len(token_text) == 1 and not token_text.isalpha():
            continue

        # Check syllable count
        if count_syllables(token_text) == syllable_count:
            filtered_indices.append(idx)

    if not filtered_indices:
        # Fallback to original logits if no matches
        return logits

    # Create new logits with filtered tokens
    new_logits = mx.full_like(logits, -float("inf"))
    for idx in filtered_indices:
        new_logits[idx] = logits[idx]

    return new_logits


async def inference_generator(
    prompt: str, max_tokens: int, use_syllable_filter: bool = False, syllable_count: int = 2
):
    """Generate tokens using mlx_lm.stream_generate with KV caching."""

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Create logits processors list
    logits_processors = []
    if use_syllable_filter:
        logits_processors.append(
            lambda tokens, logits: syllable_logits_processor(tokens, logits, syllable_count)
        )

    # Use stream_generate which handles KV caching automatically
    for response in mlx_lm.stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        logits_processors=logits_processors if logits_processors else None,
    ):
        yield response.text
        # Allow other async tasks to run
        await asyncio.sleep(0)


@app.post("/generate")
async def generate(request: PromptRequest):
    """Generate text from a prompt with streaming response."""
    if model is None:
        return {"error": "Model not loaded. Server may still be initializing."}

    async def stream_response():
        async for token in inference_generator(
            request.prompt, request.max_tokens, request.use_syllable_filter, request.syllable_count
        ):
            yield token

    return StreamingResponse(stream_response(), media_type="text/plain")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Gemma FastAPI Server")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/gemma-3-4b-it-8bit",
        help="Model to load (default: mlx-community/gemma-3-4b-it-8bit)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")

    args = parser.parse_args()

    # Load model before starting server
    load_model(args.model)

    # Start server
    uvicorn.run(app, host=args.host, port=args.port)
