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
    prompt: list
    max_tokens: Optional[int] = 512
    use_syllable_filter: Optional[bool] = False
    syllable_count: Optional[int] = 2


# use_syllable_filter and syllable_count can be passed with the request to
# explicitly constrain generated tokens to words with the given number of
# syllables.

app = FastAPI()

# Global model and tokenizer
model = None
tokenizer = None
cmu_dict = None


def top_k_logits_processor(tokens: mx.array, logits: mx.array, k: int = 50) -> mx.array:
    """Top-k logits processor that works properly with MLX."""
    # Simple approach: set all non-top-k values to -inf
    # Get the k-th largest value for each batch
    kth_largest = mx.sort(logits, axis=-1)[:, -k][:, None]  # Shape: (batch_size, 1)

    # Create mask where logits >= kth_largest
    mask = logits >= kth_largest

    # Apply the mask: keep top-k logits, set others to -inf
    filtered_logits = mx.where(mask, logits, -mx.inf)

    return filtered_logits


# Alternative approach: Dynamic syllable mapping that works with actual vocab size
class SyllableLogitsProcessor:
    """Filter logits so only tokens with the target syllable count are allowed."""

    def __init__(self, tokenizer, cmu_dict, target_syllable_count: int):
        self.tokenizer = tokenizer
        self.cmu_dict = cmu_dict
        self.target_syllable_count = target_syllable_count
        self.valid_token_mask = None

    def count_syllables(self, word: str) -> int:
        word_lower = word.lower()
        if word_lower in self.cmu_dict:
            return len([ph for ph in self.cmu_dict[word_lower][0] if ph[-1].isdigit()])
        vowels = "aeiouAEIOU"
        return sum(1 for char in word if char in vowels)

    def _build_mask(self, vocab_size: int) -> mx.array:
        mask = []
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        tokenizer_vocab_size = len(self.tokenizer.get_vocab())
        max_token = min(vocab_size, tokenizer_vocab_size)

        for token_id in range(vocab_size):
            if token_id >= max_token:
                mask.append(False)
                continue
            try:
                if eos_token_id is not None and token_id == eos_token_id:
                    mask.append(True)
                    continue

                token_text = self.tokenizer.decode([token_id]).strip()

                if token_text in [".", ",", "!", "?", "/"] or len(token_text) == 0:
                    mask.append(True)
                    continue

                if len(token_text) == 1 and not token_text.isalpha():
                    mask.append(False)
                    continue

                if self.count_syllables(token_text) == self.target_syllable_count:
                    mask.append(True)
                else:
                    mask.append(False)
            except Exception:
                mask.append(False)

        return mx.array(mask)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if self.valid_token_mask is None or self.valid_token_mask.shape[0] != logits.shape[-1]:
            self.valid_token_mask = self._build_mask(logits.shape[-1])

        filtered_logits = mx.where(self.valid_token_mask[None, :], logits, -mx.inf)

        if mx.all(mx.isinf(filtered_logits)):
            return logits

        return filtered_logits




def load_model(model_name: str = "mlx-community/gemma-3-4b-it-8bit"):
    """Load the model and tokenizer."""
    global model, tokenizer, cmu_dict
    print(f"Loading model: {model_name}")
    model, tokenizer = mlx_lm.load(model_name)

    # Initialize CMU dictionary for syllable counting
    if cmu_dict is None:
        cmu_dict = cmudict.dict()

    print("Model loaded successfully")


# Global syllable processor instances
syllable_processors = {}


def get_syllable_processor(syllable_count: int):
    """Get or create a syllable processor for the given syllable count."""
    if syllable_count not in syllable_processors:
        syllable_processors[syllable_count] = SyllableLogitsProcessor(
            tokenizer, cmu_dict, syllable_count
        )
    return syllable_processors[syllable_count]




async def inference_generator(
    prompt: list,
    max_tokens: int,
    use_syllable_filter: bool = False,
    syllable_count: int = 2,
):
    """Generate tokens with optional syllable filtering and full KV caching."""

    # Apply chat template
    messages = prompt
    print(f"Prompt: {messages}")
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"Formatted prompt: {formatted_prompt}")

    logits_processors = []
    if use_syllable_filter:
        logits_processors.append(get_syllable_processor(syllable_count))

    print("Starting generation...")

    # Single stream_generate call with full KV caching
    for response in mlx_lm.stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        logits_processors=logits_processors,
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
            request.prompt,
            request.max_tokens,
            request.use_syllable_filter,
            request.syllable_count,
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
