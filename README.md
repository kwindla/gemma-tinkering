
# install requirements

pip install -r requirements.txt

You may have dependency issues between Pipecat and MLX. Need to write down how to solve thos.

# .env looks like this
CARTESIA_API_KEY=...
DAILY_ROOM_URL=..
DAILY_TOKEN=... # hackathon room access

# run the inference server

python gemma_server.py --model mlx-community/gemma-3-12b-it-8bit

# run the bot launcher

python local-dev-server.py

# then connect from any basic RTVI client

