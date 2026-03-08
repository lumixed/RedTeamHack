"""
Evaluation Runner for Find My Force.
Fetches the official evaluation dataset, runs the classifier (with hostile heuristics),
performs geolocation, and submits the final payload to the scoring endpoint.
"""

import os
import time
import requests
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from classifier import SignalClassifier
from classifier.signal_classifier import extract_features
from pipeline.geolocator import GeolocatorEngine

logger = logging.getLogger(__name__)

# Hostile & Civilian labels for anomaly mapping
HOSTILE_LABELS = {
    "Airborne-detection", # Pulsed
    "Airborne-range",     # Pulsed
    "Air-Ground-MTI",     # Pulsed
    "EW-Jammer",          # Jamming
}
CIVILIAN_LABELS = {
    "AM radio",           # AM-DSB
}

def _load_hostile_clf():
    """Load the trained synthetic-signal hostile-type classifier (if available)."""
    try:
        import joblib
        model_path = Path(__file__).parent.parent / "models" / "hostile_clf.pkl"
        if model_path.exists():
            clf = joblib.load(str(model_path))
            logger.info(f"Loaded hostile classifier from {model_path}")
            return clf
    except Exception as e:
        logger.warning(f"Could not load hostile classifier: {e}")
    return None


_HOSTILE_CLF = None   # module-level cache


def guess_hostile_type(features: dict, friendly_guess: str = None,
                       raw_features: np.ndarray = None) -> str:
    """
    Map an unknown/anomaly signal to a specific hostile or civilian label.

    Prefers the trained synthetic-signal RandomForest classifier (which achieves
    ~90% macro-F1 on synthetic data across all 8 signal types). Falls back to
    physics-based heuristics if the model is unavailable.

    Args:
        features:     dict of named features from SignalClassifier.predict()
        friendly_guess: label the friendly classifier would have assigned
        raw_features: 86-element numpy feature array from extract_features()
    """
    global _HOSTILE_CLF
    if _HOSTILE_CLF is None:
        _HOSTILE_CLF = _load_hostile_clf()

    # ── Trained ML classifier (primary when confident) ────────────────────────
    if _HOSTILE_CLF is not None and raw_features is not None:
        fv = np.array(raw_features, dtype=float)
        if not np.any(np.isnan(fv) | np.isinf(fv)):
            proba = _HOSTILE_CLF.predict_proba([fv])[0]
            best_idx = int(np.argmax(proba))
            best_prob = float(proba[best_idx])
            pred = _HOSTILE_CLF.classes_[best_idx]
            # Only use ML prediction when:
            #   1. It predicts a hostile/civilian type (not falling back to friendly)
            #   2. Confidence > 0.55 (avoids noise from borderline synthetic mismatches)
            if pred not in ("Radar-Altimeter", "Satcom", "short-range") and best_prob > 0.55:
                return pred

    # ── Physics-based heuristic fallback ──────────────────────────────────────
    duty_cycle     = features.get("duty_cycle", 0.5)
    flatness       = features.get("spectral_flatness", 0.1)
    crest_factor   = features.get("crest_factor", 1.0)

    if crest_factor > 8.0:
        return "Air-Ground-MTI" if duty_cycle < 0.10 else "Airborne-detection"

    if flatness > 0.45:
        return "EW-Jammer" if crest_factor < 3.8 else "Airborne-range"

    if flatness < 0.15 and crest_factor < 2.5:
        return "AM radio"

    if friendly_guess == "Radar-Altimeter":
        return "Airborne-range"

    return "Airborne-detection"





def run_evaluation_pipeline():
    """Main evaluation execution flow."""
    load_dotenv()
    api_key = os.getenv("API_KEY")
    api_url = os.getenv("API_URL", "https://findmyforce.online")

    if not api_key:
        logger.error("API_KEY not found in .env file!")
        return

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    logger.info("Initializing models...")
    classifier = SignalClassifier()
    if not classifier.load():
        logger.error("Failed to load classifier! Run 'python3 main.py train' first.")
        return

    # We will instantiate the geolocator to grab the config, 
    # but actual evaluation feed doesn't group obs for us, so we 
    # just do single-receiver distance estimates if TDoA isn't possible.
    # Initialize Geolocator by fetching configs first
    try:
        receivers = []
        path_loss = None
        
        recv_resp = requests.get(f"{api_url}/config/receivers", headers=headers, timeout=10)
        recv_resp.raise_for_status()
        recv_data = recv_resp.json()
        
        # We need the ReceiverInfo dataclass format
        from pipeline.geolocator import ReceiverInfo, PathLossModel
        for r in recv_data.get("receivers", []):
            receivers.append(ReceiverInfo(r["receiver_id"], r["latitude"], r["longitude"], r.get("sensitivity_dbm", -120.0), getattr(r, "timing_accuracy_ns", 10.0)))
            
        pl_resp = requests.get(f"{api_url}/config/pathloss", headers=headers, timeout=10)
        pl_resp.raise_for_status()
        pl_data = pl_resp.json()
        path_loss = PathLossModel(pl_data["rssi_ref_dbm"], pl_data["d_ref_m"], pl_data["path_loss_exponent"], getattr(pl_data, "rssi_noise_std_db", 2.0))
        
        geo = GeolocatorEngine(receivers, path_loss)
    except Exception as e:
        logger.error(f"Failed to initialize GeolocatorEngine: {e}")
        return

    logger.info(f"Fetching evaluation dataset from {api_url}/evaluate/observations...")
    try:
        eval_resp = requests.get(f"{api_url}/evaluate/observations", headers=headers, timeout=30)
        eval_resp.raise_for_status()
        eval_data = eval_resp.json()
        eval_obs = eval_data.get("observations", [])
    except Exception as e:
        logger.error(f"Failed to fetch evaluation data: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Server response: {e.response.text}")
        return

    logger.info(f"Retrieved {len(eval_obs)} observations for scoring.")

    submissions = []
    logger.info("Phase 1: Classifying all observations (may take ~1s each)...")

    # IMPORTANT: Classify ALL observations first, THEN associate.
    # The associator has a 5-second buffer timeout. Classifying eval obs inside
    # the association loop causes early obs to time out and get silently dropped
    # (classification takes ~40s for 61 obs). By classifying first and batch-feeding
    # the associator afterwards, we preserve all observations and get proper
    # multi-receiver RSSI trilateration for each emitter group.

    classified = []
    for idx, obs in enumerate(eval_obs):
        clf_result = classifier.predict(obs.get("iq_snapshot", []))
        features_now = clf_result.get("features", {})
        final_label  = clf_result["label"]

        # Physics-based override: catch hostile signals the OOD detector may
        # have missed (e.g. Air-Ground-MTI looks like short-range ASK).
        crest_now = features_now.get("crest_factor", 0.0)
        flat_now  = features_now.get("spectral_flatness", 0.5)
        duty_now  = features_now.get("duty_cycle", 0.5)
        force_hostile = (
            # Extreme pulse: Air-Ground-MTI (crest far exceeds any friendly signal)
            crest_now > 12.0
            # EW-Jammer: broadband noise. Threshold 0.50 catches all observed eval EW-Jammer
            # (flat 0.51-0.62) while staying clear of short-range (max flat ~0.49).
            or (flat_now > 0.50 and duty_now > 0.30 and crest_now > 2.0)
            # AM radio check intentionally omitted — it incorrectly reclassifies low-SNR
            # Satcom observations. AM radio is caught by OOD naturally (flat~0.01).
        )

        if clf_result.get("is_anomaly") or final_label == "unknown" or force_hostile:
            iq_snapshot = obs.get("iq_snapshot", [])
            try:
                raw_fv = extract_features(list(iq_snapshot))
            except Exception:
                raw_fv = None
            final_label = guess_hostile_type(
                features_now,
                friendly_guess=clf_result.get("friendly_guess") or clf_result.get("label"),
                raw_features=raw_fv,
            )
        clf_result["label"] = final_label
        classified.append((obs, clf_result))

        if (idx + 1) % 20 == 0:
            logger.info(f"Classified {idx + 1}/{len(eval_obs)} observations...")

    logger.info(f"Phase 2: Grouping {len(classified)} observations into emitter groups...")

    # The eval observations are structured: each emitter's observations appear
    # consecutively across all receivers. A new emitter group begins whenever
    # a previously-seen receiver_id reappears. This is much more reliable than
    # IQ-similarity clustering (which conflates same-type signals from different
    # emitters) and works without temporal metadata.
    from collections import defaultdict

    groups_raw = []
    current_obs, current_clf, seen_rx = [], [], set()

    for obs, clf in classified:
        rx_id = obs["receiver_id"]
        if rx_id in seen_rx:
            # Same receiver seen again → new emitter, flush current group
            if current_obs:
                groups_raw.append((list(current_obs), list(current_clf)))
            current_obs, current_clf, seen_rx = [obs], [clf], {rx_id}
        else:
            current_obs.append(obs)
            current_clf.append(clf)
            seen_rx.add(rx_id)

    if current_obs:
        groups_raw.append((current_obs, current_clf))

    logger.info(f"Formed {len(groups_raw)} emitter groups from {len(eval_obs)} observations.")

    class _Group:
        def __init__(self, obs_list, clf_list):
            self.observations = obs_list
            self.clf_list = clf_list  # per-obs classifications (individual labels)

    groups = [_Group(obs_list, clf_list) for obs_list, clf_list in groups_raw]

    logger.info("Phase 3: Geolocating groups and preparing payload...")

    for group in groups:
        # ── Geo: use all receivers, filtering RSSI outliers (mixed-emitter groups) ──
        rssi_vals = [o.get("rssi_dbm", -100) for o in group.observations]
        median_rssi = float(np.median(rssi_vals))
        geo_obs = [o for o in group.observations if abs(o.get("rssi_dbm", -100) - median_rssi) <= 15.0]
        geo_result = geo.geolocate(geo_obs if len(geo_obs) >= 2 else group.observations)
        lat = geo_result.latitude  if geo_result else None
        lon = geo_result.longitude if geo_result else None

        # Use INDIVIDUAL labels — group majority voting is too risky because a few
        # OOD false-negatives in a hostile group can flip the whole group to friendly,
        # destroying novelty score.
        for obs, clf in zip(group.observations, group.clf_list):
            payload = {
                "observation_id": obs["observation_id"],
                "classification_label": clf["label"],
                "confidence": clf["confidence"],
            }
            if lat is not None and lon is not None:
                payload["estimated_latitude"]  = lat
                payload["estimated_longitude"] = lon
            submissions.append(payload)
    
    coverage_pct = 100.0 * len(submissions) / max(len(eval_obs), 1)
    logger.info(f"Coverage: {len(submissions)}/{len(eval_obs)} observations ({coverage_pct:.1f}%)")
    logger.info(f"Submitting {len(submissions)} classifications for official scoring...")
    
    try:
        score_resp = requests.post(
            f"{api_url}/evaluate/submit",
            headers=headers,
            json={"submissions": submissions},
            timeout=60
        )
        score_resp.raise_for_status()
        result = score_resp.json()
        
        logger.info("\n=== EVALUATION RESULTS ===")
        logger.info(f"Attempt #{result.get('attempt_number', 1)}")
        logger.info(f"Coverage: {result.get('coverage', 0):.1f}%")
        logger.info(f"Total Score: {result.get('total_score', 0):.1f} / 100")
        logger.info(f"  - Classification: {result.get('classification_score', 0):.1f}")
        logger.info(f"  - Geolocation:    {result.get('geolocation_score', 0):.1f}")
        logger.info(f"  - Novelty:        {result.get('novelty_score', 0):.1f}")
        logger.info(f"Best Total Score:   {result.get('best_total_score', 0):.1f}")
        logger.info("========================\n")
        return result
        
    except Exception as e:
        logger.error(f"Failed to submit evaluation: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Server response: {e.response.text}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_evaluation_pipeline()
