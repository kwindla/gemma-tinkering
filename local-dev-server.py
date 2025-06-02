import os
from typing import Dict, Any
from fastapi import HTTPException
from fastapi import Request
import subprocess
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import argparse
import json

import dotenv

dotenv.load_dotenv()


DAILY_ROOM_URL = os.getenv("DAILY_ROOM_URL")
DAILY_TOKEN = os.getenv("DAILY_TOKEN")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/connect")
async def rtvi_connect(request: Request) -> Dict[Any, Any]:
    """RTVI connect endpoint that creates a room and returns connection credentials.

    This endpoint is called by RTVI clients to establish a connection.

    Returns:
        Dict[Any, Any]: Authentication bundle containing room_url and token

    Raises:
        HTTPException: If room creation, token generation, or bot startup fails
    """
    print("Returning room for RTVI connection")
    room_url, token = (DAILY_ROOM_URL, DAILY_TOKEN)
    print(f"Room URL: {room_url}")
    body = await request.json()
    print(f"Body: {body}")

    # Start the bot process
    try:
        bot_file = "bot.py"
        body_escaped = json.dumps(body)
        print(f"Body escaped: {body_escaped}")
        proc = subprocess.Popen(
            [f"python3 {bot_file} '{body_escaped}'"],
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    # Return the authentication bundle in format expected by DailyTransport
    return {"room_url": room_url, "token": token}


if __name__ == "__main__":
    import uvicorn

    # Parse command line arguments for server configuration
    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("FAST_API_PORT", "7860"))

    parser = argparse.ArgumentParser(description="Daily Storyteller FastAPI server")
    parser.add_argument("--host", type=str, default=default_host, help="Host address")
    parser.add_argument("--port", type=int, default=default_port, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Reload code on change")

    config = parser.parse_args()

    # Start the FastAPI server
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        reload=config.reload,
    )
