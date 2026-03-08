import os
import requests
import json
import sseclient
import numpy as np
from classifier.signal_classifier import SignalClassifier
from dotenv import load_dotenv

load_dotenv()
clf = SignalClassifier()
if not clf.load():
    print("Model not loaded")
    exit(1)

url = "https://findmyforce.online/feed/stream"
headers = {"X-API-Key": os.getenv("API_KEY"), "Accept": "text/event-stream"}
response = requests.get(url, headers=headers, stream=True)
client = sseclient.SSEClient(response)

count = 0
for event in client.events():
    if event.event == "observation":
        data = json.loads(event.data)
        rx_id = data.get("receiver_id")
        iq = data.get("iq_snapshot", [])
        if iq:
            res = clf.predict(iq)
            print(f"Obs: {data['observation_id']} | RX: {rx_id} | Label: {res['label']} | Anomaly: {res['is_anomaly']}")
            count += 1
            if count >= 10:
                break
