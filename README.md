
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

# notes

mlx-community/gemma-3-4b-it-8bit produces a four-syllable word for this generation request:

```
curl -X POST http://localhost:8000/generate \
    -H 'Content-Type: application/json' \
    -d '{"prompt": [{"role": "user", "content": "Write a four-line poem about computer programming."}], "use_syllable_filter":true, "syllable_count":2}'
```

output:

```
Within circuits, logic resides,
Commands crafted, neatly applied.
Building software, pixelated sprites,
Coding futures, shining, brilliant, delights.
```

Notice "pixelated". Why? Because pixelated is actually two tokens. Our logit sampler properly limits the output to two-syllable tokens, but then we concate "pixel" and "ated" into a single, english-language, word!

