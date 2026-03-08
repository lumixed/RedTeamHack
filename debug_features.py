import os
import requests
import json
import sseclient
import numpy as np
from classifier.signal_classifier import SignalClassifier, extract_features
from pipeline.eval_runner import guess_hostile_type
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

print("Monitoring for anomalies...")
count = 0
for event in client.events():
    if event.event == "observation":
        data = json.loads(event.data)
        iq = data.get("iq_snapshot", [])
        if iq:
            res = clf.predict(iq)
            if res['is_anomaly'] or res['label'] == 'unknown':
                feat = res['features']
                guess = guess_hostile_type(feat)
                print(f"Anomaly! Guess: {guess} | OOD: {res['ood_score']:.2f}")
                print(f"  Duty: {feat['duty_cycle']:.3f} | Flat: {feat['spectral_flatness']:.3f} | Lin: {feat['freq_linearity']:.3f}")
                print(f"  FreqStd: {feat['freq_std']:.3f} | PAPR: {feat['papr']:.3f} | PhaseJump: {feat['phase_jumps_180']:.3f}")
                print(f"  AmpStd: {feat['amp_std']:.3f} | TotalPwr: {feat['total_power']:.3f}")
                count += 1
                if count >= 5:
                    break
