
# install requirements

pip install -r requirements.txt

You may have dependency issues between Pipecat and MLX. Need to write down how to solve thos.

# .env looks like this
CARTESIA_API_KEY=...
DAILY_ROOM_URL=..
DAILY_TOKEN=... # hackathon room access

# run the inference server

python -m gemma_server --model mlx-community/gemma-3-12b-it-8bit

# run inference from the command line

python -m gemma_server --model mlx-community/gemma-3-12b-it-8bit \
    --prompt "Tell me a haiku" --max-tokens 20

# generate using curl

curl -X POST http://localhost:8000/generate \
    -H 'Content-Type: application/json' \
    -d '{"prompt": [{"role": "user", "content": "Tell me a haiku"}], "max_tokens": 20}'

# run the bot launcher

python local-dev-server.py

# then connect from any basic RTVI client

