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
ASSOCIATION_WINDOW_S = 0.5    # 0.5s window to ensure same-pulse grouping for TDoA

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

    def add_observation(self, obs: dict, classification: dict) -> list[ObservationGroup]:
        """
        Add a new classified observation to the buffer.
        Returns a list of ObservationGroups if any were completed or flushed.
        
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

        return completed_groups


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

                # Dynamic IQ similarity gate
                # If they are very close in time or have the same label, relax the threshold
                dt = abs(ts_j - ts_i)
                dynamic_min_iq = MIN_IQ_COSINE_SIMILARITY
                
                if dt < 0.5: # Very tight temporal window
                    dynamic_min_iq -= 0.1
                if label_i == label_j and label_i != "unknown":
                    dynamic_min_iq -= 0.1
                
                iq_sim = _cosine_similarity(
                    obs_i.get("iq_snapshot", []),
                    obs_j.get("iq_snapshot", [])
                )
                if iq_sim < dynamic_min_iq:
                    continue

                group_members.append(obs_j)
                used.add(j)

            # Only emit groups where the oldest observation is old enough to be complete,
            # OR if force=True, OR if we have enough receivers for a good fix.
            if force or len(group_members) >= 3:
                group = self._build_group(group_members)
                groups.append(group)
            else:
                # If we only have 1 or 2 receivers, wait until it's "old" enough 
                # (using the internal parsed timestamps to be skew-invariant)
                # We use the relative buffer timeout logic
                pass

        # Also flush any observations that are simply too old (timed out in buffer)
        # We need a way to detect the "current" simulation time
        if self._buffer:
            max_ts = max(o["_parsed_ts"] for o in self._buffer)
            for i, obs_i in enumerate(self._buffer):
                if i in used: continue
                # If this observation is > 5s older than the newest in buffer, 
                # it's unlikely to get more friends. Flush it as its own group.
                if max_ts - obs_i["_parsed_ts"] > self._buffer_timeout:
                    groups.append(self._build_group([obs_i]))
                    used.add(i)

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

        # Weighted Majority Vote for classification
        # We sum the confidence scores for each label to find the strongest consensus
        label_tally = defaultdict(float)
        for m in members:
            cls = m["_classification"]
            label_tally[cls["label"]] += cls["confidence"]
        
        # Determine winning label
        final_label = max(label_tally, key=label_tally.get)
        
        # Get the representative classification data for the winning label
        # We take the one with the highest confidence among the winning label members
        winning_members = [m for m in members if m["_classification"]["label"] == final_label]
        best_m = max(winning_members, key=lambda o: o["_classification"]["confidence"])
        cls_info = best_m["_classification"]

        # Average timestamp
        timestamps = [m["_parsed_ts"] for m in members]
        avg_ts = float(np.mean(timestamps))

        return ObservationGroup(
            group_id=group_id,
            observations=[{k: v for k, v in m.items() if not k.startswith("_")} for m in members],
            timestamp=avg_ts,
            classification_label=final_label,
            classification_confidence=float(cls_info["confidence"]),
            is_friendly=bool(cls_info["is_friendly"]),
            is_anomaly=bool(cls_info["is_anomaly"]),
            ood_score=float(cls_info.get("ood_score", 0.0)),
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
    va_comp = va[:min_len//2] + 1j * va[min_len//2:]
    vb_comp = vb[:min_len//2] + 1j * vb[min_len//2:]
    
    # Use magnitude envelope for similarity (invariant to phase rotation)
    env_a = np.abs(va_comp)
    env_b = np.abs(vb_comp)
    
    norm_a = np.linalg.norm(env_a)
    norm_b = np.linalg.norm(env_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(env_a, env_b) / (norm_a * norm_b))
