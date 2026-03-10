"""
Find My Force - Dashboard API Server
Flask + SocketIO real-time web server for the Common Operating Picture (COP).
Serves the frontend and exposes REST + WebSocket endpoints for live track data.
"""

import os
import sys
import json
import time
import logging
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory, render_template_string
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Setup paths
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

from pipeline import (
    GeolocatorEngine, ReceiverInfo, PathLossModel,
    TrackManager, TrackUpdate,
    ObservationAssociator,
    FeedConsumer, EvalSubmitter, get_config, get_score,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask App ─────────────────────────────────────────────────────────────────
# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__, 
            static_folder=str(ROOT_DIR / "dashboard" / "static"),
            static_url_path='/static')
app.config["SECRET_KEY"] = os.urandom(24)
CORS(app)

# =========================================================================
# Socket.IO Handlers
# =========================================================================
# Use gevent or eventlet if available, fallback to threading
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")

# ── Global State ──────────────────────────────────────────────────────────────
# Built during initialization rather than at import. Constructing it here would
# pull torch in before the socket is bound, which on a small vCPU keeps the port
# closed long enough for a proxy to give up on the app.
g_classifier = None
g_track_manager = TrackManager()
g_associator = ObservationAssociator()
g_geolocator: Optional[GeolocatorEngine] = None
g_feed_consumer: Optional[FeedConsumer] = None
g_eval_submitter: Optional[EvalSubmitter] = None
g_receiver_config = None
g_pathloss_config = None
g_recent_observations = []  # Ring buffer of last 100 observations
g_max_recent = 100
g_server_status = {
    "pipeline_running": False,
    "classifier_trained": False,
    "training_in_progress": False,
    "training_metrics": None,
    "error": None,
    "api_url": os.getenv("API_URL", "https://findmyforce.online"),
    "api_key_set": bool(os.getenv("API_KEY", "")),
}


# ── Initialization ─────────────────────────────────────────────────────────────
_initialized = False


def _fetch_config(attempts=90, delay=2.0):
    """Poll the range for receiver and path-loss config until it answers."""
    for attempt in range(attempts):
        try:
            rx_data, pl_data = get_config()
            if rx_data:
                return rx_data, pl_data
        except Exception as e:
            logger.debug(f"Config fetch failed: {e}")
        if attempt == 0:
            logger.info("Range not answering yet, waiting for it to come up")
        time.sleep(delay)

    logger.error(f"Range never returned a receiver config after {attempts} attempts")
    return None, None


def initialize_system(force=False):
    """
    Initialize the full pipeline:
    1. Fetch receiver config & path loss from API
    2. Load classifier
    3. Start feed consumer (if not in serverless mode)
    """
    global g_geolocator, g_feed_consumer, g_eval_submitter, _initialized
    global g_receiver_config, g_pathloss_config, g_classifier

    if _initialized and not force:
        return

    from classifier import SignalClassifier
    if g_classifier is None:
        g_classifier = SignalClassifier()

    logger.info("=== Initializing Find My Force Pipeline ===")

    # 1. Fetch server config. Keep asking rather than accepting the first refusal:
    # this runs while the range may still be starting, and giving up here leaves
    # the pipeline with no receivers for the rest of the process lifetime, able to
    # classify but never to geolocate.
    rx_data, pl_data = _fetch_config(attempts=90, delay=2.0)

    if rx_data:
        g_receiver_config = rx_data
        receivers = [
            ReceiverInfo(
                receiver_id=r["receiver_id"],
                latitude=r["latitude"],
                longitude=r["longitude"],
                sensitivity_dbm=r.get("sensitivity_dbm", -90.0),
                timing_accuracy_ns=r.get("timing_accuracy_ns", 50.0),
            )
            for r in rx_data.get("receivers", [])
        ]
        logger.info(f"Loaded {len(receivers)} receivers")
    else:
        logger.warning("Could not fetch receiver config - using empty config")
        receivers = []

    if pl_data:
        g_pathloss_config = pl_data
        pathloss = PathLossModel(
            rssi_ref_dbm=pl_data.get("rssi_ref_dbm", -30.0),
            d_ref_m=pl_data.get("d_ref_m", 1.0),
            path_loss_exponent=pl_data.get("path_loss_exponent", 2.8),
            rssi_noise_std_db=pl_data.get("rssi_noise_std_db", 3.5),
        )
    else:
        logger.warning("Could not fetch path loss config - using defaults")
        pathloss = PathLossModel(-30.0, 1.0, 2.8, 3.5)

    # Set ref position from first receiver
    if receivers:
        g_track_manager._ref_lat = receivers[0].latitude
        g_track_manager._ref_lon = receivers[0].longitude

    # 2. Initialize geolocator
    g_geolocator = GeolocatorEngine(receivers, pathloss)

    # 3. Try to load existing classifier (Essential for prediction)
    try:
        model_loaded = g_classifier.load()
        if model_loaded:
            g_server_status["classifier_trained"] = True
            logger.info("Pre-trained classifier loaded from disk")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

    # 4. Initialize eval submitter
    g_eval_submitter = EvalSubmitter(g_classifier, g_geolocator)

    # 5. Start feed consumer (ONLY if not on Vercel/Serverless)
    # Vercel doesn't allow background processes/SSE listeners
    is_vercel = os.getenv("VERCEL") == "1"
    
    if not is_vercel:
        def on_track_update(track_dict, geo_result):
            socketio.emit("track_update", {
                "track": track_dict,
                "all_tracks": g_track_manager.get_all_as_dict(),
                "stats": g_track_manager.get_stats(),
            })

        def on_observation(obs):
            global g_recent_observations
            light_obs = {
                "observation_id": obs.get("observation_id"),
                "receiver_id": obs.get("receiver_id"),
                "rssi_dbm": obs.get("rssi_dbm"),
                "snr_estimate_db": obs.get("snr_estimate_db"),
                "timestamp": obs.get("timestamp"),
                "classification": obs.get("_classification"),
            }
            g_recent_observations.append(light_obs)
            if len(g_recent_observations) > g_max_recent:
                g_recent_observations = g_recent_observations[-g_max_recent:]
            socketio.emit("observation", light_obs)

        g_feed_consumer = FeedConsumer(
            classifier=g_classifier,
            associator=g_associator,
            geolocator=g_geolocator,
            track_manager=g_track_manager,
            on_track_update=on_track_update,
            on_observation=on_observation,
        )
        g_feed_consumer.start()
        
        # Periodic loops
        def submission_loop():
            while True:
                try:
                    if g_feed_consumer: g_feed_consumer.submit_queued()
                except: pass
                time.sleep(2.0)
        
        def broadcast_loop():
            while True:
                try:
                    socketio.emit("tracks_broadcast", {
                        "tracks": g_track_manager.get_all_as_dict(),
                        "stats": g_track_manager.get_stats(),
                        "feed_stats": g_feed_consumer.stats if g_feed_consumer else {},
                    })
                except: pass
                time.sleep(5.0)

        threading.Thread(target=submission_loop, daemon=True).start()
        threading.Thread(target=broadcast_loop, daemon=True).start()
        
        g_server_status["pipeline_running"] = True
        logger.info("Background pipeline started")
    else:
        logger.info("Serverless mode: Background pipeline disabled")
        g_server_status["pipeline_running"] = False

    _initialized = True

_init_thread = None
_init_lock = threading.Lock()


def start_initialization():
    """Kick off initialization once, in the background."""
    global _init_thread
    with _init_lock:
        if _init_thread is None:
            _init_thread = threading.Thread(target=initialize_system, daemon=True)
            _init_thread.start()


@app.before_request
def ensure_initialized():
    """
    Make sure initialization is under way, without waiting for it. Running it
    inline here blocks every request for as long as the range takes to come up,
    which reads to a proxy as an app that accepts connections but never answers.
    """
    if not _initialized:
        start_initialization()


# ── REST API Endpoints ─────────────────────────────────────────────────────────

@app.route("/api/health")
def api_health():
    return jsonify({"status": "healthy", "initialized": _initialized, "serverless": os.getenv("VERCEL") == "1"})

@app.route("/api/status")
def api_status():
    """System status endpoint."""
    health = None
    try:
        import requests as req
        r = req.get(f"{g_server_status['api_url']}/health", timeout=2)
        health = r.json() if r.status_code == 200 else None
    except Exception:
        pass

    return jsonify({
        "system": g_server_status,
        "server_health": health,
        "tracks": g_track_manager.get_stats() if g_track_manager else {},
        "feed_stats": g_feed_consumer.stats if g_feed_consumer else {},
        "receiver_config": g_receiver_config,
        "pathloss_config": g_pathloss_config,
    })


@app.route("/api/tracks")
def api_tracks():
    """Return all active tracks."""
    return jsonify({
        "tracks": g_track_manager.get_all_as_dict() if g_track_manager else [],
        "stats": g_track_manager.get_stats() if g_track_manager else {},
    })


@app.route("/api/score/fetch", methods=["GET"])
def fetch_official_score():
    """Fetch current official score from the backend API."""
    import requests
    api_key = os.getenv("API_KEY")
    api_url = os.getenv("API_URL", "https://findmyforce.online")
    if not api_key:
        return jsonify({"error": "Missing API Key"}), 401
    try:
        resp = requests.get(f"{api_url}/scores/me", headers={"X-API-Key": api_key}, timeout=5)
        return (jsonify(resp.json()), resp.status_code) if resp.ok else (jsonify({"error": resp.reason}), resp.status_code)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    """Train classifier on uploaded HDF5 data (Disabled on Vercel)."""
    if os.getenv("VERCEL") == "1":
        return jsonify({"error": "Training is disabled in serverless mode. Train locally and push the model."}), 405
    
    global g_server_status
    if g_server_status["training_in_progress"]:
        return jsonify({"error": "Training already in progress"}), 409

    data_dir = ROOT_DIR / "data"
    hdf5_files = list(data_dir.glob("*.h5")) + list(data_dir.glob("*.hdf5"))
    if not hdf5_files:
        return jsonify({"error": "No HDF5 training file found in /data directory"}), 404

    hdf5_path = str(hdf5_files[0])
    g_server_status["training_in_progress"] = True

    def train_bg():
        try:
            from classifier import load_training_data
            X_feat, X_raw, y = load_training_data(hdf5_path)
            metrics = g_classifier.train(X_feat, X_raw, y)
            g_classifier.save()
            g_server_status["classifier_trained"] = True
            g_server_status["training_metrics"] = metrics
            socketio.emit("training_complete", {"metrics": metrics})
        except Exception as e:
            logger.error(f"Training error: {e}")
            g_server_status["error"] = str(e)
            socketio.emit("training_error", {"error": str(e)})
        finally:
            g_server_status["training_in_progress"] = False

    threading.Thread(target=train_bg, daemon=True).start()
    return jsonify({"message": "Training started", "file": hdf5_path})


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """Classify a single IQ snapshot."""
    body = request.get_json()
    if not body or "iq_snapshot" not in body:
        return jsonify({"error": "No iq_snapshot provided"}), 400
    if g_classifier is None:
        return jsonify({"error": "Classifier still loading"}), 503
    return jsonify(g_classifier.predict(body["iq_snapshot"]))


@app.route("/api/observations")
def api_observations():
    """Return recent observations."""
    return jsonify({"observations": g_recent_observations[-50:] if g_recent_observations else []})


@app.route("/api/score")
def api_score_legacy():
    """Alias for score fetch."""
    score = get_score()
    return jsonify(score or {"error": "Could not fetch score"})


@app.route("/api/eval/run", methods=["POST"])
def api_eval_run():
    """Run the evaluation submission."""
    if not g_eval_submitter:
        return jsonify({"error": "System not initialized"}), 503

    # On Vercel, we must run it synchronously or it will be killed
    if os.getenv("VERCEL") == "1":
        result = g_eval_submitter.run_eval()
        return jsonify({"result": result})
    
    def run_bg():
        result = g_eval_submitter.run_eval()
        socketio.emit("eval_complete", {"result": result})

    threading.Thread(target=run_bg, daemon=True).start()
    return jsonify({"message": "Evaluation submission started"})


@app.route("/api/receivers")
def api_receivers():
    """Return receiver configurations."""
    return jsonify(g_receiver_config or {"receivers": []})


# ── Frontend Serving ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main COP dashboard."""
    return send_from_directory(str(ROOT_DIR / "dashboard"), "index.html")

@app.route("/dashboard/<path:filename>")
def dashboard_static_serve(filename):
    return send_from_directory(str(ROOT_DIR / "dashboard"), filename)

# ── SocketIO Events ────────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    emit("init", {
        "tracks": g_track_manager.get_all_as_dict() if g_track_manager else [],
        "stats": g_track_manager.get_stats() if g_track_manager else {},
        "status": g_server_status,
        "receivers": g_receiver_config,
    })

@socketio.on("request_tracks")
def on_request_tracks():
    emit("tracks_broadcast", {
        "tracks": g_track_manager.get_all_as_dict() if g_track_manager else [],
        "stats": g_track_manager.get_stats() if g_track_manager else {},
        "feed_stats": g_feed_consumer.stats if g_feed_consumer else {},
    })

@socketio.on("request_eval")
def on_request_eval():
    if os.getenv("VERCEL") == "1":
        result = g_eval_submitter.run_eval()
        emit("eval_complete", {"result": result})
        return
    
    def run_bg():
        try:
            from pipeline.eval_runner import run_evaluation_pipeline
            result = run_evaluation_pipeline()
            socketio.emit("eval_complete", {"result": result})
        except Exception as exc:
            logger.error(f"Eval pipeline error: {exc}")
            socketio.emit("eval_complete", {"result": None, "error": str(exc)})
    threading.Thread(target=run_bg, daemon=True).start()
    emit("eval_started", {})


# ── Main ──────────────────────────────────────────────────────────────────────
# Nothing is initialized at import. Callers start initialize_system() on a thread
# and bind their socket straight away; it waits on the range, and blocking the
# import on that would hold the listening port closed for as long as the range
# takes to come up.
if __name__ == "__main__":
    start_initialization()
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Starting dashboard on http://localhost:{port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
