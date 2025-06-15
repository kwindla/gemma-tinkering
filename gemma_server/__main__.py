import argparse
import asyncio
import uvicorn

from .app import app
from .inference import inference_generator, load_model


def main():
    parser = argparse.ArgumentParser(
        description="Gemma inference server and CLI. If --prompt is provided, inference runs on the command line; otherwise the HTTP server starts."
    )
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
    parser.add_argument(
        "--prompt",
        type=str,
        help="If provided, run CLI generation instead of starting the server",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate in CLI mode",
    )
    parser.add_argument(
        "--use-syllable-filter",
        action="store_true",
        help="Apply syllable filter during generation",
    )
    parser.add_argument(
        "--syllable-count",
        type=int,
        default=2,
        help="Syllable count for the syllable filter",
    )

    args = parser.parse_args()
    load_model(args.model)

    if args.prompt:

        async def run_cli():
            prompt = [{"role": "user", "content": args.prompt}]
            async for token in inference_generator(
                prompt,
                args.max_tokens,
                args.use_syllable_filter,
                args.syllable_count,
            ):
                print(token, end="", flush=True)

        asyncio.run(run_cli())
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
