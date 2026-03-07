"""
Live Feed Consumer
Connects to the Find My Force API, streams observations,
classifies them, associates them, geolocates, and updates tracks.
Also handles submission to the scoring API.
"""

import os
import time
import json
import logging
import threading
import requests
from collections import deque
from typing import Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

# API constants
API_URL = os.getenv("API_URL", "https://findmyforce.online")
API_KEY = os.getenv("API_KEY", "")

# Rate limiting
SUBMIT_COOLDOWN_S = 1.0  # Min seconds between submissions
EVAL_SUBMIT_COOLDOWN_S = 60.0  # Min seconds between eval submissions


class FeedConsumer:
    """
    Connects to the SSE feed and processes observations end-to-end.
    Coordinates: classifier → associator → geolocator → track manager → submission
    """

    def __init__(
        self,
        classifier,
        associator,
        geolocator,
        track_manager,
        on_track_update: Optional[Callable] = None,
        on_observation: Optional[Callable] = None,
    ):
        self.classifier = classifier
        self.associator = associator
        self.geolocator = geolocator
        self.track_manager = track_manager
        self.on_track_update = on_track_update
        self.on_observation = on_observation

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_submit_time = 0.0
        self._submission_queue: deque = deque(maxlen=500)
        self._submitted_ids: set = set()

        # Stats
        self.stats = {
            "observations_received": 0,
            "groups_formed": 0,
            "tracks_updated": 0,
            "submissions_sent": 0,
            "errors": 0,
            "start_time": None,
        }

    def start(self):
        """Start the feed consumer in a background thread."""
        self._running = True
        self.stats["start_time"] = time.time()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Feed consumer started")

    def stop(self):
        """Stop the feed consumer."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Feed consumer stopped")

    def _run_loop(self):
        """Main processing loop - reconnects on failure."""
        while self._running:
            try:
                self._process_sse_stream()
            except Exception as e:
                logger.error(f"Feed error: {e}")
                self.stats["errors"] += 1
                if self._running:
                    time.sleep(3.0)

    def _process_sse_stream(self):
        """Connect to SSE stream and process events."""
        logger.info(f"Connecting to SSE stream at {API_URL}/feed/stream")
        headers = {"X-API-Key": API_KEY, "Accept": "text/event-stream"}

        with requests.get(
            f"{API_URL}/feed/stream",
            headers=headers,
            stream=True,
            timeout=60,
        ) as resp:
            if resp.status_code == 401:
                logger.error("Authentication failed - check API_KEY")
                time.sleep(10)
                return
            elif resp.status_code != 200:
                logger.error(f"Feed returned {resp.status_code}")
                time.sleep(5)
                return

            logger.info("SSE stream connected")
            for line in resp.iter_lines(decode_unicode=True):
                if not self._running:
                    break
                if line and line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str:
                        try:
                            self._process_observation(json.loads(data_str))
                        except json.JSONDecodeError:
                            pass
                        except Exception as e:
                            logger.warning(f"Error processing observation: {e}")
                            self.stats["errors"] += 1

    def _process_observation(self, obs: dict):
        """Process a single observation through the pipeline."""
        self.stats["observations_received"] += 1

        # 1. Classify the signal
        classification = self.classifier.predict(obs.get("iq_snapshot", []))
        
        # Apply heuristic mapping for anomalies ONLY if the classifier didn't find a known label
        final_label = classification.get("label", "unknown")
        is_known = final_label in {"Radar-Altimeter", "Satcom", "short-range", "AM radio", "FM radio", "LTE", "WiFi"}
        
        if (classification.get("is_anomaly") and not is_known) or final_label == "unknown":
            from pipeline.eval_runner import guess_hostile_type
            final_label = guess_hostile_type(
                classification.get("features", {}),
                friendly_guess=classification.get("friendly_guess"),
            )
            classification["label"] = final_label

        # 2. Enrich observation
        obs["_classification"] = classification

        if self.on_observation:
            try:
                self.on_observation(obs)
            except Exception:
                pass
        
        # 3. Update simulation clock
        from .associator import _parse_timestamp
        self.track_manager.update_clock(obs.get("_parsed_ts", 0.0) or _parse_timestamp(obs.get("timestamp", "")) or 0.0)

        # 4. Associate with other observations
        groups = self.associator.add_observation(obs, classification)

        for group in groups:
            self.stats["groups_formed"] += 1
            self._process_group(group)


    def _process_group(self, group):
        """Process an observation group through geolocation and track management."""
        # Geolocation
        geo_result = self.geolocator.geolocate(group.observations)

        if not geo_result:
            if len(group.observations) > 1:
                logger.info(f"Geolocation failed for {group.classification_label} group ({len(group.observations)} obs)")
            return

        # Track update
        import numpy as np
        from .track_manager import TrackUpdate
        update = TrackUpdate(
            timestamp=group.timestamp,
            latitude=self._safe_float(geo_result.latitude),
            longitude=self._safe_float(geo_result.longitude),
            uncertainty_m=self._safe_float(geo_result.uncertainty_m),
            classification_label=group.classification_label,
            confidence=self._safe_float(group.classification_confidence, 0.5),
            n_receivers=geo_result.n_receivers,
            method=geo_result.method,
            observation_ids=[o.get("observation_id", "") for o in group.observations],
            rssi_dbm=self._safe_float(float(np.mean([o.get("rssi_dbm", -80) for o in group.observations])), -80.0),
            snr_db=self._safe_float(float(np.mean([o.get("snr_estimate_db", 0) for o in group.observations])), 0.0),
        )
        track_id = self.track_manager.update(update)
        self.stats["tracks_updated"] += 1

        track = next(
            (t for t in self.track_manager.all_tracks if t.track_id == track_id),
            None
        )

        if self.on_track_update and track:
            try:
                self.on_track_update(track.to_dict(), geo_result)
            except Exception:
                pass

        # Submit all observations using the group's consensus label + smoothed track position
        # (group.observations have _-prefixed keys stripped by the associator)
        group_cls = {
            "label": group.classification_label,
            "confidence": group.classification_confidence,
        }
        lat = track.latitude if track else geo_result.latitude
        lon = track.longitude if track else geo_result.longitude
        for obs in group.observations:
            self._queue_submission(obs, group_cls, lat, lon)


    def _safe_float(self, val, default=None):
        if val is None: return default
        try:
            f = float(val)
            import math
            if math.isnan(f) or math.isinf(f): return default
            return f
        except:
            return default

    def _queue_submission(self, obs: dict, classification: dict,
                          lat: float = None, lon: float = None):
        """Queue an observation for submission to the API."""
        obs_id = obs.get("observation_id", "")
        if not obs_id or obs_id in self._submitted_ids:
            return

        label = classification.get("label", "unknown")
        confidence = self._safe_float(classification.get("confidence"), 0.5)
        lat = self._safe_float(lat)
        lon = self._safe_float(lon)

        self._submission_queue.append({
            "observation_id": obs_id,
            "classification_label": label,
            "confidence": confidence,
            "estimated_latitude": lat,
            "estimated_longitude": lon,
        })

    def submit_queued(self):
        """Submit queued observations to the API (call periodically)."""
        now = time.time()
        if now - self._last_submit_time < SUBMIT_COOLDOWN_S:
            return

        submitted = 0
        while self._submission_queue:
            item = self._submission_queue.popleft()
            obs_id = item["observation_id"]

            if obs_id in self._submitted_ids:
                continue

            payload = {
                "observation_id": obs_id,
                "classification_label": item["classification_label"],
                "confidence": item["confidence"],
            }
            if item.get("estimated_latitude") is not None:
                payload["estimated_latitude"] = item["estimated_latitude"]
                payload["estimated_longitude"] = item["estimated_longitude"]

            try:
                resp = requests.post(
                    f"{API_URL}/submissions/classify",
                    headers={
                        "X-API-Key": API_KEY,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=5,
                )
                if resp.status_code == 200:
                    self._submitted_ids.add(obs_id)
                    self.stats["submissions_sent"] += 1
                    submitted += 1
                elif resp.status_code == 429:
                    # Rate limited - put back and wait
                    self._submission_queue.appendleft(item)
                    time.sleep(2.0)
                    break
            except Exception as e:
                logger.warning(f"Submission error: {e}")

            time.sleep(SUBMIT_COOLDOWN_S * 0.1)

        self._last_submit_time = now
        if submitted > 0:
            logger.debug(f"Submitted {submitted} observations")


class EvalSubmitter:
    """
    Handles the evaluation endpoint:
    Fetches eval observations, classifies them, and submits for scoring.
    """

    def __init__(self, classifier, geolocator):
        self.classifier = classifier
        self.geolocator = geolocator
        self._last_eval_submit = 0.0
        self._best_score = 0.0

    def run_eval(self) -> Optional[dict]:
        """Fetch eval observations, classify, and submit. Returns submission result."""
        now = time.time()
        if now - self._last_eval_submit < EVAL_SUBMIT_COOLDOWN_S:
            wait = EVAL_SUBMIT_COOLDOWN_S - (now - self._last_eval_submit)
            logger.info(f"Eval submission cooldown: {wait:.0f}s remaining")
            return None

        # Check if eval is open
        health = self._check_health()
        if not health or not health.get("evaluation_open", False):
            logger.info("Evaluation is not yet open")
            return None

        # Fetch eval observations
        observations = self._fetch_eval_observations()
        if not observations:
            logger.warning("No eval observations returned")
            return None

        logger.info(f"Fetched {len(observations)} eval observations")

        # Classify each
        submissions = []
        for obs in observations:
            iq = obs.get("iq_snapshot", [])
            classification = self.classifier.predict(iq)

            payload = {
                "observation_id": obs["observation_id"],
                "classification_label": classification["label"],
                "confidence": classification["confidence"],
            }

            # Try single-receiver geolocation
            geo = self.geolocator.geolocate([obs])
            if geo:
                payload["estimated_latitude"] = geo.latitude
                payload["estimated_longitude"] = geo.longitude

            submissions.append(payload)

        # Submit
        result = self._submit_eval(submissions)
        if result:
            self._last_eval_submit = now
            logger.info(f"Eval submitted: {result}")

        return result

    def _check_health(self) -> Optional[dict]:
        try:
            resp = requests.get(f"{API_URL}/health", timeout=5)
            return resp.json() if resp.status_code == 200 else None
        except Exception:
            return None

    def _fetch_eval_observations(self) -> list[dict]:
        try:
            resp = requests.get(
                f"{API_URL}/evaluate/observations",
                headers={"X-API-Key": API_KEY},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("observations", data if isinstance(data, list) else [])
            logger.warning(f"Eval obs fetch failed: {resp.status_code} {resp.text[:200]}")
            return []
        except Exception as e:
            logger.error(f"Error fetching eval observations: {e}")
            return []

    def _submit_eval(self, submissions: list[dict]) -> Optional[dict]:
        try:
            resp = requests.post(
                f"{API_URL}/evaluate/submit",
                headers={
                    "X-API-Key": API_KEY,
                    "Content-Type": "application/json",
                },
                json={"submissions": submissions},
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"Eval submit failed: {resp.status_code} {resp.text[:200]}")
            return None
        except Exception as e:
            logger.error(f"Error submitting eval: {e}")
            return None


def get_score() -> Optional[dict]:
    """Fetch team score from the API."""
    try:
        resp = requests.get(
            f"{API_URL}/scores/me",
            headers={"X-API-Key": API_KEY},
            timeout=10,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def get_config() -> tuple[Optional[dict], Optional[dict]]:
    """Fetch receiver config and path loss model from the API."""
    headers = {"X-API-Key": API_KEY}
    try:
        rx_resp = requests.get(f"{API_URL}/config/receivers", headers=headers, timeout=10)
        pl_resp = requests.get(f"{API_URL}/config/pathloss", headers=headers, timeout=10)
        receivers = rx_resp.json() if rx_resp.status_code == 200 else None
        pathloss = pl_resp.json() if pl_resp.status_code == 200 else None
        return receivers, pathloss
    except Exception as e:
        logger.error(f"Error fetching config: {e}")
        return None, None


try:
    import numpy as np
except ImportError:
    pass
