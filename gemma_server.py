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


app = FastAPI()

# Global model and tokenizer
model = None
tokenizer = None
cmu_dict = None


def syllable_logits_processor(tokens: mx.array, logits: mx.array, syllable_count: int) -> mx.array:
    """
    Logits processor that filters tokens by syllable count.
    This version works with MLX's tensor operations.
    """
    # For MLX compatibility, we need to precompute which tokens are valid
    # and store them in a way that doesn't require Python loops during inference

    # Simple approach: use a pre-computed mask or fall back to top-k filtering
    # since syllable filtering requires tokenizer.decode() which breaks MLX graphs

    return top_k_logits_processor(tokens, logits, k=50)


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
    """Class-based processor that dynamically builds syllable mappings."""

    def __init__(self, tokenizer, cmu_dict, target_syllable_count: int):
        self.tokenizer = tokenizer
        self.cmu_dict = cmu_dict
        self.target_syllable_count = target_syllable_count
        self.valid_token_mask = None  # Will be built on first call

    def _build_syllable_mask(self, vocab_size: int):
        """Build syllable mask based on actual vocabulary size."""
        # Build mask as a Python list first, then convert to MLX array
        valid_token_list = []

        # Get EOS token ID for special handling
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        # Only check up to the tokenizer's known vocabulary size
        tokenizer_vocab_size = len(self.tokenizer.get_vocab())
        max_token_to_check = min(vocab_size, tokenizer_vocab_size)

        print(f"EOS token ID: {eos_token_id}")

        # This runs once during first inference call
        for token_id in range(vocab_size):
            if token_id < max_token_to_check:
                try:
                    # Always allow EOS token
                    if eos_token_id is not None and token_id == eos_token_id:
                        valid_token_list.append(True)
                        continue

                    token_text = self.tokenizer.decode([token_id]).strip()

                    # Always allow special tokens and whitespace
                    if token_text in ["", "/", " "] or len(token_text) == 0:
                        valid_token_list.append(True)
                        continue

                    # Skip single non-alphabetic characters
                    if len(token_text) == 1 and not token_text.isalpha():
                        valid_token_list.append(False)
                        continue

                    # Check syllable count
                    if self.count_syllables(token_text) == self.target_syllable_count:
                        valid_token_list.append(True)
                    else:
                        valid_token_list.append(False)

                except Exception:
                    # Skip problematic tokens
                    valid_token_list.append(False)
            else:
                # Tokens beyond tokenizer vocabulary - default to False
                valid_token_list.append(False)

        # Convert to MLX array
        return mx.array(valid_token_list)

    def count_syllables(self, word: str) -> int:
        """Count syllables in a word using CMU dictionary."""
        word_lower = word.lower()
        if word_lower in self.cmu_dict:
            return len([ph for ph in self.cmu_dict[word_lower][0] if ph[-1].isdigit()])
        # Fallback: count vowels as rough approximation
        vowels = "aeiouAEIOU"
        return sum(1 for char in word if char in vowels)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """Apply syllable filtering using dynamically-built mask."""
        # Build mask on first call based on actual logits shape
        if self.valid_token_mask is None:
            vocab_size = logits.shape[-1]
            print(f"Building syllable mask for vocab size: {vocab_size}")
            self.valid_token_mask = self._build_syllable_mask(vocab_size)

        # Ensure mask matches logits shape
        if self.valid_token_mask.shape[0] != logits.shape[-1]:
            print(
                f"Mask size mismatch: {self.valid_token_mask.shape[0]} vs {logits.shape[-1]}, rebuilding..."
            )
            vocab_size = logits.shape[-1]
            self.valid_token_mask = self._build_syllable_mask(vocab_size)

        # Apply the mask
        filtered_logits = mx.where(
            self.valid_token_mask[None, :],  # Broadcast across batch dimension
            logits,
            -mx.inf,
        )

        # If no tokens are valid (all -inf), fall back to top-k
        if mx.all(mx.isinf(filtered_logits)):
            return top_k_logits_processor(tokens, logits, k=20)

        return filtered_logits


# Updated server code modifications:
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
    """Generate tokens using mlx_lm.stream_generate with KV caching."""

    # Apply chat template
    messages = prompt
    print(f"Prompt: {messages}")
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"Formatted prompt: {formatted_prompt}")

    # Create logits processors list
    logits_processors = []
    if use_syllable_filter:
        processor = get_syllable_processor(syllable_count)
        logits_processors.append(processor)

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
