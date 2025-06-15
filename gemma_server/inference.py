"""Model loading and inference utilities for the Gemma server."""

import asyncio
from typing import List

import mlx_lm
import cmudict

from .logits_processors import SyllableLogitsProcessor

# Global model and tokenizer
model = None
tokenizer = None
cmu_dict = None

# Cache of syllable processors
syllable_processors = {}


def load_model(model_name: str = "mlx-community/gemma-3-4b-it-8bit"):
    """Load the model and tokenizer."""
    global model, tokenizer, cmu_dict
    print(f"Loading model: {model_name}")
    model, tokenizer = mlx_lm.load(model_name)
    if cmu_dict is None:
        cmu_dict = cmudict.dict()
    print("Model loaded successfully")


def get_syllable_processor(syllable_count: int):
    """Get or create a syllable processor for the given syllable count."""
    if syllable_count not in syllable_processors:
        syllable_processors[syllable_count] = SyllableLogitsProcessor(
            tokenizer, cmu_dict, syllable_count
        )
    return syllable_processors[syllable_count]


async def inference_generator(
    prompt: List,
    max_tokens: int,
    use_syllable_filter: bool = False,
    syllable_count: int = 2,
):
    """Generate tokens with optional syllable filtering and full KV caching."""

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

    for response in mlx_lm.stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        logits_processors=logits_processors,
    ):
        yield response.text
        await asyncio.sleep(0)
