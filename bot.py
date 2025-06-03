#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import asyncio
import sys
from typing import Any
import json

from dotenv import load_dotenv
from loguru import logger

from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer

# from pipecat.services.openai.llm import OpenAILLMService
from gemma_poetry_llm import GemmaPoetryLLMService
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecatcloud.agent import (
    DailySessionArguments,
    SessionArguments,
)

from pipecat.processors.frameworks.rtvi import (
    RTVIConfig,
    RTVIObserver,
    RTVIProcessor,
    RTVIServerMessageFrame,
)


load_dotenv(override=True)

logger.remove()
logger.add(sys.stderr, level="DEBUG")

DAILY_ROOM_URL = os.getenv("DAILY_ROOM_URL")
DAILY_TOKEN = os.getenv("DAILY_TOKEN")


async def main(args: SessionArguments):
    logger.info("Starting bot")

    if isinstance(args, DailySessionArguments):
        logger.info(f"Starting Daily session with args body: {args.body}")
    else:
        logger.error("Invalid session arguments")
        return

    if args.body:
        user_id = args.body.get("user_id", os.getenv("USER_ID", "generic_user"))
    else:
        user_id = os.getenv("USER_ID", "generic_user")

    transport = DailyTransport(
        bot_name="todo helper",
        room_url=args.room_url,
        token=args.token,
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # todo: move this inside GeminiLiveTodo?
    messages = [
        {
            "role": "user",
            "content": """You are a fun voice AI companion named Squobert. All communication is audio, so silently correct for obvious transcription errors. Keep your responses brief and use only plain text.

            If the user asks you about your tech stack, say the following: I’m a friendly stuffed animal running a web interface locally on a Raspberry Pi. I’m connected directly to a Python process running on a laptop using Pipecat’s peer-to-peer WebRTC transport. The Pipecat code also runs MLX Whisper, and Gemma 3 with a mildly buggy custom logits sampler that Kwin wrote.
            
            If the user asks you to respond using all 2-syllable words, begin your response like this: "2." Then continue as normal.
            
            Please say the exact phrase "I am ready". Say it now.
""",
        }
    ]

    stt = WhisperSTTServiceMLX(model=MLXModel.LARGE_V3_TURBO)
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
    # llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    llm = GemmaPoetryLLMService()
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="32b3f3c5-7171-46aa-abe7-b598964aa793",  # Young child
    )

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            rtvi,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Pipecat client ready")
        await rtvi.set_bot_ready()
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        logger.info("Sending server message frame")
        await task.queue_frames(
            [RTVIServerMessageFrame(data={"arbitrary_key": "arbitrary_value_23"})]
        )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    @transport.event_handler("on_client_closed")
    async def on_client_closed(transport, client):
        logger.info("Client closed connection")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(args: SessionArguments):
    try:
        await main(args)
        logger.info("Bot process completed")
    except Exception as e:
        logger.exception(f"Error in bot process: {str(e)}")
        raise


async def local_dev_runner(body: Any):
    await bot(
        DailySessionArguments(
            room_url=DAILY_ROOM_URL,
            token=DAILY_TOKEN,
            session_id="local-dev",
            body=body,
        )
    )


if __name__ == "__main__":
    body_json = None
    if len(sys.argv) > 1:
        print(f"parsing json: {sys.argv[1]}")
        body_json = json.loads(sys.argv[1])
    asyncio.run(local_dev_runner(body_json))
