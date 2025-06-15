"""FastAPI application for serving Gemma model inference."""

import asyncio
from typing import Optional, List

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .inference import inference_generator, load_model, model

app = FastAPI()


class PromptRequest(BaseModel):
    prompt: List
    max_tokens: Optional[int] = 512
    use_syllable_filter: Optional[bool] = False
    syllable_count: Optional[int] = 2


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


def init(model_name: str):
    """Load the model before starting the server."""
    load_model(model_name)
