import os
import requests
import json
import sseclient
import time
from dotenv import load_dotenv

load_dotenv()
url = "https://findmyforce.online/feed/stream"
headers = {"X-API-Key": os.getenv("API_KEY"), "Accept": "text/event-stream"}
response = requests.get(url, headers=headers, stream=True)
client = sseclient.SSEClient(response)

start_obs = None
start_wall = time.time()

for event in client.events():
    if event.event == "observation":
        data = json.loads(event.data)
        ts = data.get("timestamp")
        print(f"Obs: {data['observation_id']} | TS: {ts} | Wall: {time.strftime('%H:%M:%S', time.gmtime())}")
        break
