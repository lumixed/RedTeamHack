"""
Stand-in for the hackathon range API.

Serves the same routes the pipeline already talks to, so pointing API_URL at
this process is the only change needed to run the whole system offline.
Because the simulator knows ground truth, submissions are scored for real
rather than acknowledged blindly.
"""

import json
import queue
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from flask import Flask, Response, jsonify, request

from .scenario import (
    Scenario, RECEIVERS, PATH_LOSS, SENSITIVITY_DBM, TIMING_ACCURACY_NS,
)
from .signals import FRIENDLY_TYPES, SIGNAL_TYPES

# Deliberately not imported from the classifier package: that module pulls in
# torch, and the range never runs inference. Loading it here cost the simulator
# several seconds of startup and a few hundred MB for a set of three strings.
FRIENDLY_LABELS = set(FRIENDLY_TYPES)

logger = logging.getLogger(__name__)

TEAM_NAME = "LockedIn"
EVAL_EMISSIONS = 12
TICK_S = 0.5

# Recent observations replayed to each new subscriber. A pipeline starting from
# cold otherwise needs about half a minute of live feed before the map holds
# anything, which is the whole first impression on a deployed demo. Sized to a
# couple of emissions per emitter - enough to confirm tracks, small enough that
# the classifier drains the backlog in seconds.
BACKLOG_OBSERVATIONS = 120

# Scenario time played out at startup so the backlog is already populated when the
# first subscriber arrives. On a cold container both processes start together, so
# without this there is no history to replay and the map takes half a minute to
# fill. Emissions are backdated to land just before the present.
PREROLL_S = 25.0


class FeedHub:
    """Runs the scenario and fans observations out to SSE subscribers."""

    def __init__(self, n_emitters=8, seed=None):
        self.scenario = Scenario(n_emitters=n_emitters, seed=seed)
        self.subscribers = []
        self.backlog = deque(maxlen=BACKLOG_OBSERVATIONS)
        self.truth = {}
        self.lock = threading.Lock()
        self.live_submissions = {}
        self.eval_attempts = []
        self.best_total = 0.0
        self.started = time.time()
        self.emitted = 0
        self._eval = None
        self._eval_lock = threading.Lock()

    def eval_data(self):
        """Built on first use; keeping it off the startup path speeds up binding."""
        with self._eval_lock:
            if self._eval is None:
                self._eval = self._build_eval_set()
        return self._eval

    def _build_eval_set(self):
        """A fixed held-out set, emitters kept consecutive as the eval runner assumes."""
        observations, truth = [], {}
        scratch = Scenario(n_emitters=len(SIGNAL_TYPES), seed=97)
        for index in range(EVAL_EMISSIONS):
            emitter = scratch.emitters[index % len(scratch.emitters)]
            scratch.advance(1.0)
            for obs in scratch.observations_for(emitter):
                truth[obs["observation_id"]] = {
                    "label": emitter.signal_type,
                    "latitude": emitter.latitude,
                    "longitude": emitter.longitude,
                }
                observations.append(obs)
        return observations, truth

    def start(self):
        # Prefill runs inside the worker rather than before it, so the port binds
        # immediately. Generating a scenario takes long enough on a small vCPU that
        # doing it first leaves the platform's proxy knocking on a closed socket.
        threading.Thread(target=self._run, daemon=True).start()

    def prefill(self):
        """Play the scenario forward into the backlog before accepting subscribers."""
        now = datetime.now(timezone.utc)
        end = self.scenario.clock + PREROLL_S

        while self.scenario.clock < end:
            self.scenario.advance(TICK_S)
            behind = max(end - self.scenario.clock, 0.0)
            stamp = (now - timedelta(seconds=behind)).isoformat().replace("+00:00", "Z")
            for emitter in self.scenario.due():
                for obs in self.scenario.observations_for(emitter, timestamp=stamp):
                    self.truth[obs["observation_id"]] = {
                        "label": emitter.signal_type,
                        "latitude": emitter.latitude,
                        "longitude": emitter.longitude,
                    }
                    self.emitted += 1
                    self._publish(obs)

        logger.info(f"Pre-rolled {PREROLL_S:.0f}s of scenario, {len(self.backlog)} observations buffered")

    def _run(self):
        self.prefill()
        while True:
            self.scenario.advance(TICK_S)
            for emitter in self.scenario.due():
                stamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                for obs in self.scenario.observations_for(emitter, timestamp=stamp):
                    with self.lock:
                        self.truth[obs["observation_id"]] = {
                            "label": emitter.signal_type,
                            "latitude": emitter.latitude,
                            "longitude": emitter.longitude,
                        }
                        self.emitted += 1
                    self._publish(obs)
            time.sleep(TICK_S)

    def _publish(self, obs):
        payload = f"data: {json.dumps(obs)}\n\n"
        with self.lock:
            self.backlog.append(payload)
            targets = list(self.subscribers)
        for q in targets:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass

    def subscribe(self):
        q = queue.Queue(maxsize=BACKLOG_OBSERVATIONS + 400)
        with self.lock:
            for payload in self.backlog:
                q.put_nowait(payload)
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


def _haversine_m(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * 6371000.0 * asin(sqrt(a))


def score_submissions(submissions, truth):
    """
    Score a batch the way the original brief weighted it:
    classification 40, geolocation 30, novelty 30.
    """
    matched = [(s, truth[s["observation_id"]]) for s in submissions
               if s.get("observation_id") in truth]
    if not matched:
        return None

    correct = sum(1 for s, t in matched if s.get("classification_label") == t["label"])
    classification = 40.0 * correct / len(matched)

    distances = [
        _haversine_m(s["estimated_latitude"], s["estimated_longitude"],
                     t["latitude"], t["longitude"])
        for s, t in matched
        if s.get("estimated_latitude") is not None and s.get("estimated_longitude") is not None
    ]
    if distances:
        distances.sort()
        cep = distances[len(distances) // 2]
        geolocation = 30.0 * max(0.0, 1.0 - min(cep, 2000.0) / 2000.0)
    else:
        cep = None
        geolocation = 0.0

    novel = [(s, t) for s, t in matched if t["label"] not in FRIENDLY_LABELS]
    if novel:
        flagged = sum(1 for s, t in novel
                      if s.get("classification_label") not in FRIENDLY_LABELS)
        novelty = 30.0 * flagged / len(novel)
    else:
        novelty = 0.0

    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "count": 0})
    for s, t in matched:
        predicted = s.get("classification_label")
        per_class[t["label"]]["count"] += 1
        if predicted == t["label"]:
            per_class[t["label"]]["tp"] += 1
        else:
            per_class[t["label"]]["fn"] += 1
            per_class[predicted]["fp"] += 1

    per_class_scores = []
    for label, c in sorted(per_class.items()):
        denom = 2 * c["tp"] + c["fp"] + c["fn"]
        per_class_scores.append({
            "label": label,
            "f1": round(2.0 * c["tp"] / denom, 3) if denom else 0.0,
            "count": c["count"],
        })

    return {
        "matched": len(matched),
        "classification_score": round(classification, 1),
        "geolocation_score": round(geolocation, 1),
        "novelty_score": round(novelty, 1),
        "total_score": round(classification + geolocation + novelty, 1),
        "average_cep_meters": round(cep, 1) if cep is not None else None,
        "per_class_scores": per_class_scores,
    }


def create_app(hub):
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({
            "status": "healthy",
            "evaluation_open": True,
            "uptime_s": round(time.time() - hub.started, 1),
            "observations_emitted": hub.emitted,
        })

    @app.get("/config/receivers")
    def config_receivers():
        return jsonify({
            "receivers": [
                {**r,
                 "sensitivity_dbm": SENSITIVITY_DBM,
                 "timing_accuracy_ns": TIMING_ACCURACY_NS}
                for r in RECEIVERS
            ]
        })

    @app.get("/config/pathloss")
    def config_pathloss():
        return jsonify(PATH_LOSS)

    @app.get("/feed/stream")
    def feed_stream():
        q = hub.subscribe()

        def emit():
            try:
                while True:
                    try:
                        yield q.get(timeout=15.0)
                    except queue.Empty:
                        yield ": keepalive\n\n"
            finally:
                hub.unsubscribe(q)

        return Response(emit(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/submissions/classify")
    def submit_classify():
        body = request.get_json(silent=True) or {}
        obs_id = body.get("observation_id")
        if not obs_id:
            return jsonify({"error": "observation_id required"}), 400
        with hub.lock:
            hub.live_submissions[obs_id] = body
        return jsonify({"accepted": True, "observation_id": obs_id})

    @app.get("/evaluate/observations")
    def eval_observations():
        observations, _ = hub.eval_data()
        return jsonify({"observations": observations})

    @app.post("/evaluate/submit")
    def eval_submit():
        body = request.get_json(silent=True) or {}
        submissions = body.get("submissions", [])
        observations, truth = hub.eval_data()
        result = score_submissions(submissions, truth)
        if result is None:
            return jsonify({"error": "no scorable submissions"}), 400

        hub.eval_attempts.append(result)
        hub.best_total = max(hub.best_total, result["total_score"])
        return jsonify({
            "attempt_number": len(hub.eval_attempts),
            "coverage": round(100.0 * result["matched"] / max(len(observations), 1), 1),
            "total_score": result["total_score"],
            "classification_score": result["classification_score"],
            "geolocation_score": result["geolocation_score"],
            "novelty_score": result["novelty_score"],
            "best_total_score": hub.best_total,
        })

    @app.get("/scores/me")
    def scores_me():
        with hub.lock:
            submissions = list(hub.live_submissions.values())
            truth = dict(hub.truth)
        result = score_submissions(submissions, truth) or {}
        return jsonify({
            "team_name": TEAM_NAME,
            "total_score": result.get("total_score", 0.0),
            "classification_score": result.get("classification_score", 0.0),
            "geolocation_score": result.get("geolocation_score", 0.0),
            "novelty_detection_score": result.get("novelty_score", 0.0),
            "submissions_count": len(submissions),
            "average_cep_meters": result.get("average_cep_meters"),
            "per_class_scores": result.get("per_class_scores", []),
        })

    return app


def serve(port=5051, n_emitters=8, seed=None):
    # One access line per submission at roughly nine submissions a second buries
    # everything else in the log.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    hub = FeedHub(n_emitters=n_emitters, seed=seed)
    hub.start()
    app = create_app(hub)
    logger.info(f"Simulated range API listening on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False, use_reloader=False)
