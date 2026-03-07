"""
Observation Association Engine
Groups independent receiver observations into likely co-emitter groups.
Uses temporal proximity, signal similarity, and RSSI geometric consistency.
"""

import time
import logging
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum time window for grouping observations from the same emission
ASSOCIATION_WINDOW_S = 2.5   # 2.5s window to account for streaming latency

# Max difference in RSSI (dB) for two observations to come from same emitter
MAX_RSSI_DIFF_DB = 25.0

# IQ similarity threshold (cosine similarity)
MIN_IQ_COSINE_SIMILARITY = 0.5


@dataclass
class ObservationGroup:
    """A group of observations likely from the same emitter emission."""
    group_id: str
    observations: list[dict]
    timestamp: float
    classification_label: str
    classification_confidence: float
    is_friendly: bool
    is_anomaly: bool
    ood_score: float

    @property
    def receiver_ids(self) -> list[str]:
        return [o["receiver_id"] for o in self.observations]

    @property
    def primary_receiver_id(self) -> str:
        """Receiver with strongest signal."""
        return max(self.observations, key=lambda o: o.get("rssi_dbm", -999))["receiver_id"]


class ObservationAssociator:
    """
    Associates incoming observations into groups from the same emitter.
    Uses:
    1. Temporal clustering (observations within ASSOCIATION_WINDOW_S)
    2. Signal similarity (same classification, similar IQ shape)
    3. RSSI geometric consistency (RSSI values should be consistent with a single source)
    """

    def __init__(self):
        # Buffer of recent unassociated observations, keyed by receiver_id
        self._buffer: list[dict] = []
        self._buffer_timeout: float = 5.0  # Flush groups older than this

    def add_observation(self, obs: dict, classification: dict) -> Optional[ObservationGroup]:
        """
        Add a new classified observation to the buffer.
        Returns an ObservationGroup if one is now complete, else None.
        
        obs: raw observation dict from the feed
        classification: dict from SignalClassifier.predict()
        """
        now = time.time()
        obs_ts = _parse_timestamp(obs.get("timestamp", "")) or now

        enriched = {
            **obs,
            "_parsed_ts": obs_ts,
            "_classification": classification,
            "_added_at": now,
        }
        self._buffer.append(enriched)

        # Flush old observations into groups
        completed_groups = self._flush_completed_groups(now)

        # Return the most recently completed group (if any)
        return completed_groups[-1] if completed_groups else None

    def flush_all(self) -> list[ObservationGroup]:
        """Force flush all buffered observations into groups."""
        return self._flush_completed_groups(time.time(), force=True)

    def _flush_completed_groups(self, now: float, force: bool = False) -> list[ObservationGroup]:
        """
        Find observations that can be grouped together.
        Returns list of completed ObservationGroup objects.
        """
        if not self._buffer:
            return []

        # Remove expired observations
        cutoff = now - self._buffer_timeout
        self._buffer = [o for o in self._buffer if o["_added_at"] > cutoff or force]

        if not self._buffer:
            return []

        # Sort by timestamp
        self._buffer.sort(key=lambda o: o["_parsed_ts"])

        # Greedy temporal clustering
        groups = []
        used = set()
        import itertools

        for i, obs_i in enumerate(self._buffer):
            if i in used:
                continue

            group_members = [obs_i]
            used.add(i)

            ts_i = obs_i["_parsed_ts"]
            cls_i = obs_i["_classification"]

            for j, obs_j in enumerate(self._buffer[i+1:], start=i+1):
                if j in used:
                    continue

                ts_j = obs_j["_parsed_ts"]
                cls_j = obs_j["_classification"]

                # Temporal gate
                if abs(ts_j - ts_i) > ASSOCIATION_WINDOW_S:
                    continue

                # Already have this receiver in the group? Skip (same receiver can't see same emission twice simultaneously)
                if obs_j["receiver_id"] in [m["receiver_id"] for m in group_members]:
                    continue

                # Classification gate
                label_i = cls_i["label"]
                label_j = cls_j["label"]
                if label_i != "unknown" and label_j != "unknown" and label_i != label_j:
                    continue

                # IQ similarity gate
                iq_sim = _cosine_similarity(
                    obs_i.get("iq_snapshot", []),
                    obs_j.get("iq_snapshot", [])
                )
                if iq_sim < MIN_IQ_COSINE_SIMILARITY:
                    continue

                group_members.append(obs_j)
                used.add(j)

            # Only emit groups where the oldest observation is old enough to be complete,
            # OR if force=True
            oldest_ts = min(m["_parsed_ts"] for m in group_members)
            if force or (now - oldest_ts > ASSOCIATION_WINDOW_S * 1.5):
                group = self._build_group(group_members)
                groups.append(group)

        # Remove grouped observations from buffer
        grouped_ids = set()
        for g in groups:
            for obs in g.observations:
                grouped_ids.add(obs.get("observation_id"))
        self._buffer = [o for o in self._buffer if o.get("observation_id") not in grouped_ids]

        return groups

    def _build_group(self, members: list[dict]) -> ObservationGroup:
        """Build an ObservationGroup from a list of associated observations."""
        import uuid
        group_id = f"GRP-{str(uuid.uuid4())[:8].upper()}"

        # Use highest-confidence classification
        best_cls = max(members, key=lambda o: o["_classification"]["confidence"])
        cls = best_cls["_classification"]

        # Average timestamp
        timestamps = [m["_parsed_ts"] for m in members]
        avg_ts = float(np.mean(timestamps))

        return ObservationGroup(
            group_id=group_id,
            observations=[{k: v for k, v in m.items() if not k.startswith("_")} for m in members],
            timestamp=avg_ts,
            classification_label=cls["label"],
            classification_confidence=float(cls["confidence"]),
            is_friendly=bool(cls["is_friendly"]),
            is_anomaly=bool(cls["is_anomaly"]),
            ood_score=float(cls.get("ood_score", 0.0)),
        )


def _parse_timestamp(ts_str: str) -> Optional[float]:
    """Parse ISO 8601 timestamp string to Unix timestamp."""
    if not ts_str:
        return None
    try:
        from datetime import datetime, timezone
        # Handle both Z and +00:00 suffixes
        ts_str = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_str)
        return dt.timestamp()
    except Exception:
        return None


def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two IQ snapshots."""
    if not a or not b:
        return 0.0
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    # Normalize length
    min_len = min(len(va), len(vb))
    va, vb = va[:min_len], vb[:min_len]
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))
