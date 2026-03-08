"""
Track Manager
Maintains persistent emitter tracks over time.
Handles track creation, update, association, aging, and loss.
"""

import time
import uuid
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

from .geolocator import GeoResult, KalmanTracker, latlon_to_xy, xy_to_latlon

logger = logging.getLogger(__name__)

# Track state machine
TRACK_STATES = {
    "TENTATIVE": "tentative",      # New, not yet confirmed
    "CONFIRMED": "confirmed",      # Confirmed track
    "COASTING": "coasting",        # No recent observations
    "LOST": "lost",                # Track has been dropped
}

# Affiliation categories
AFFILIATION = {
    "FRIENDLY": "friendly",
    "HOSTILE": "hostile",
    "UNKNOWN": "unknown",
    "CIVILIAN": "civilian",
}

FRIENDLY_LABELS = {"Radar-Altimeter", "Satcom", "short-range"}


@dataclass
class TrackUpdate:
    """A single update applied to a track."""
    timestamp: float
    latitude: float
    longitude: float
    uncertainty_m: float
    classification_label: str
    confidence: float
    n_receivers: int
    method: str
    observation_ids: list[str]
    rssi_dbm: float = -80.0
    snr_db: float = 0.0


@dataclass
class EmitterTrack:
    """Represents a persistent track for a single emitter."""
    track_id: str
    created_at: float
    last_seen: float
    state: str
    affiliation: str

    # Current best estimates
    latitude: float
    longitude: float
    uncertainty_m: float
    classification_label: str
    classification_confidence: float
    geolocation_method: str
    n_receivers: int

    # History
    position_history: list[dict] = field(default_factory=list)
    classification_history: list[dict] = field(default_factory=list)
    observation_count: int = 0
    update_count: int = 0

    # Kalman filter state
    kalman: Optional[KalmanTracker] = field(default=None, repr=False)
    _ref_lat: float = field(default=49.26, repr=False)
    _ref_lon: float = field(default=-123.25, repr=False)

    def to_dict(self) -> dict:
        """Serialize track to dict for API/frontend consumption."""
        age_seconds = time.time() - self.last_seen
        is_stale = age_seconds > 30.0

        return {
            "track_id": self.track_id,
            "state": self.state,
            "affiliation": self.affiliation,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "uncertainty_m": self.uncertainty_m,
            "classification_label": self.classification_label,
            "classification_confidence": self.classification_confidence,
            "geolocation_method": self.geolocation_method,
            "n_receivers": self.n_receivers,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "age_seconds": round(age_seconds, 1),
            "is_stale": is_stale,
            "observation_count": self.observation_count,
            "update_count": self.update_count,
            "position_history": self.position_history[-20:],  # Last 20 positions
            "velocity_mps": self._get_velocity(),
        }

    def _get_velocity(self) -> Optional[dict]:
        if self.kalman is None:
            return None
        vx, vy = self.kalman.velocity
        speed = np.sqrt(vx**2 + vy**2)
        return {"vx_mps": round(vx, 2), "vy_mps": round(vy, 2), "speed_mps": round(speed, 2)}


class TrackManager:
    """
    Manages a collection of emitter tracks.
    Handles association, creation, update, and lifecycle management.
    """

    # Tuning parameters
    MAX_ASSOCIATION_DISTANCE_M = 800.0    # Max position distance for track association
    MAX_ASSOCIATION_TIME_S = 15.0         # Max time gap for association
    CONFIRMATION_UPDATES = 2              # Updates needed to confirm track
    COAST_TIMEOUT_S = 30.0               # Time before track goes to COASTING
    LOST_TIMEOUT_S = 180.0                # Keep tracks alive longer to prevent fragmentation
    MAX_HISTORY = 50                      # Position history entries to keep

    def __init__(self, ref_lat: float = 49.26, ref_lon: float = -123.25):
        self._tracks: dict[str, EmitterTrack] = {}
        self._ref_lat = ref_lat
        self._ref_lon = ref_lon
        self._latest_ts = 0.0

    @property
    def active_tracks(self) -> list[EmitterTrack]:
        """Return non-lost tracks."""
        return [t for t in self._tracks.values() if t.state != TRACK_STATES["LOST"]]

    @property
    def all_tracks(self) -> list[EmitterTrack]:
        return list(self._tracks.values())

    def update_clock(self, ts: float):
        """Update the internal simulation clock."""
        if ts > self._latest_ts:
            self._latest_ts = ts

    def update(self, update: TrackUpdate) -> str:
        """
        Process a track update.
        Returns track_id of the updated/created track.
        """
        if update.timestamp > self._latest_ts:
            self._latest_ts = update.timestamp
        
        now = self._latest_ts or time.time()

        # Try to associate with existing track
        track = self._find_best_match(update)

        if track is None:
            # Create new tentative track
            track = self._create_track(update)
            logger.info(f"CREATED new track {track.track_id} for {update.classification_label} at {update.latitude}, {update.longitude}")
        else:
            logger.debug(f"Updating track {track.track_id} with new observation")

        # Update the track
        self._apply_update(track, update, now)

        return track.track_id

    def _find_best_match(self, update: TrackUpdate) -> Optional[EmitterTrack]:
        """Find the best existing track to associate this update with."""
        best_track = None
        best_score = float("inf")

        for track in self.active_tracks:
            # Time gate
            time_gap = update.timestamp - track.last_seen
            if abs(time_gap) > self.MAX_ASSOCIATION_TIME_S:
                continue

            # Velocity-aided position gate
            # Use Kalman filter to predict where the track SHOULD be now
            if time_gap > 0 and track.kalman:
                vx, vy = track.kalman.velocity
                # Current local coords
                x, y = latlon_to_xy(track.latitude, track.longitude, self._ref_lat, self._ref_lon)
                # Predict forward
                x_pred = x + vx * time_gap
                y_pred = y + vy * time_gap
                lat_pred, lon_pred = xy_to_latlon(x_pred, y_pred, self._ref_lat, self._ref_lon)
                
                dist = self._haversine_distance(
                    update.latitude, update.longitude,
                    lat_pred, lon_pred
                )
            else:
                dist = self._haversine_distance(
                    update.latitude, update.longitude,
                    track.latitude, track.longitude
                )

            if dist > self.MAX_ASSOCIATION_DISTANCE_M:
                continue

            # Classification gate (relaxed - allow unknown to match friendly)
            label_match = (
                update.classification_label == track.classification_label or
                update.classification_label == "unknown" or
                track.classification_label == "unknown"
            )

            # Score: lower = better association
            # Combine distance and classification similarity
            label_penalty = 0.0 if label_match else 500.0
            score = dist + label_penalty + abs(time_gap) * 10.0

            if score < best_score:
                best_score = score
                best_track = track

        return best_track

    def _create_track(self, update: TrackUpdate) -> EmitterTrack:
        """Create a new tentative track."""
        track_id = f"TRK-{str(uuid.uuid4())[:8].upper()}"
        now = time.time()

        # Initialize Kalman filter
        x, y = latlon_to_xy(update.latitude, update.longitude, self._ref_lat, self._ref_lon)
        kalman = KalmanTracker(x, y, init_uncertainty=update.uncertainty_m)

        affiliation = self._determine_affiliation(update.classification_label)

        track = EmitterTrack(
            track_id=track_id,
            created_at=now,
            last_seen=now,
            state=TRACK_STATES["TENTATIVE"],
            affiliation=affiliation,
            latitude=update.latitude,
            longitude=update.longitude,
            uncertainty_m=update.uncertainty_m,
            classification_label=update.classification_label,
            classification_confidence=update.confidence,
            geolocation_method=update.method,
            n_receivers=update.n_receivers,
            kalman=kalman,
            _ref_lat=self._ref_lat,
            _ref_lon=self._ref_lon,
        )

        self._tracks[track_id] = track
        return track

    def _apply_update(self, track: EmitterTrack, update: TrackUpdate, now: float):
        """Apply a track update with Kalman filtering."""
        track.observation_count += len(update.observation_ids)
        track.update_count += 1
        # Kalman predict + update
        prev_seen = track.last_seen
        track.last_seen = update.timestamp or now

        # Calculate time delta for Kalman (important: use PREVOUS last_seen)
        dt = track.last_seen - prev_seen if track.update_count > 1 else 1.0
        dt = max(0.1, min(dt, 30.0))
        track.kalman.predict(dt)

        x_meas, y_meas = latlon_to_xy(
            update.latitude, update.longitude,
            self._ref_lat, self._ref_lon
        )
        track.kalman.update(x_meas, y_meas, update.uncertainty_m)

        # Use Kalman-smoothed position
        x_smooth, y_smooth = track.kalman.position
        lat_smooth, lon_smooth = xy_to_latlon(x_smooth, y_smooth, self._ref_lat, self._ref_lon)

        track.latitude = round(lat_smooth, 6)
        track.longitude = round(lon_smooth, 6)
        track.uncertainty_m = round(track.kalman.position_uncertainty, 1)
        track.geolocation_method = update.method
        track.n_receivers = max(track.n_receivers, update.n_receivers)

        # Update classification using exponential moving average of confidence
        alpha = 0.3  # Weight for new vs. old classification
        if update.classification_label != "unknown" or track.classification_label == "unknown":
            if update.classification_label == track.classification_label:
                # Reinforce
                new_conf = (1 - alpha) * track.classification_confidence + alpha * update.confidence
            else:
                # New label competing with old - use higher confidence
                if update.confidence > track.classification_confidence:
                    track.classification_label = update.classification_label
                    new_conf = update.confidence
                else:
                    new_conf = track.classification_confidence
            track.classification_confidence = round(new_conf, 3)
        track.affiliation = self._determine_affiliation(track.classification_label)

        # Record history
        track.position_history.append({
            "timestamp": update.timestamp or now,
            "latitude": track.latitude,
            "longitude": track.longitude,
            "uncertainty_m": track.uncertainty_m,
        })
        if len(track.position_history) > self.MAX_HISTORY:
            track.position_history = track.position_history[-self.MAX_HISTORY:]

        track.classification_history.append({
            "timestamp": update.timestamp or now,
            "label": update.classification_label,
            "confidence": update.confidence,
        })
        if len(track.classification_history) > self.MAX_HISTORY:
            track.classification_history = track.classification_history[-self.MAX_HISTORY:]

        # State transitions
        if track.state == TRACK_STATES["TENTATIVE"] and track.update_count >= self.CONFIRMATION_UPDATES:
            track.state = TRACK_STATES["CONFIRMED"]
            logger.info(f"Track {track.track_id} CONFIRMED as {track.classification_label}")
        elif track.state == TRACK_STATES["COASTING"]:
            track.state = TRACK_STATES["CONFIRMED"]

    def age_tracks(self):
        """Age out stale and lost tracks based on time since last update."""
        # Use latest observation time as the 'now' for aging.
        # If no updates seen yet, don't age tracks (prevents immediate stale on startup)
        if self._latest_ts == 0.0:
            return
            
        now = self._latest_ts
        for track in list(self._tracks.values()):
            age = now - track.last_seen
            if track.state == TRACK_STATES["LOST"]:
                # Remove very old lost tracks
                if age > 300.0:
                    del self._tracks[track.track_id]
            elif track.state in [TRACK_STATES["CONFIRMED"], TRACK_STATES["TENTATIVE"]]:
                if age > self.LOST_TIMEOUT_S:
                    track.state = TRACK_STATES["LOST"]
                    logger.info(f"Track {track.track_id} LOST")
                elif age > self.COAST_TIMEOUT_S:
                    track.state = TRACK_STATES["COASTING"]

    def get_all_as_dict(self) -> list[dict]:
        """Return all non-lost tracks as serializable dicts."""
        self.age_tracks()
        return [t.to_dict() for t in self.active_tracks]

    def get_stats(self) -> dict:
        """Return track statistics."""
        tracks = list(self._tracks.values())
        active = [t for t in tracks if t.state != TRACK_STATES["LOST"]]
        return {
            "total_tracks": len(tracks),
            "active_tracks": len(active),
            "confirmed": sum(1 for t in active if t.state == TRACK_STATES["CONFIRMED"]),
            "tentative": sum(1 for t in active if t.state == TRACK_STATES["TENTATIVE"]),
            "coasting": sum(1 for t in active if t.state == TRACK_STATES["COASTING"]),
            "friendly": sum(1 for t in active if t.affiliation == AFFILIATION["FRIENDLY"]),
            "hostile": sum(1 for t in active if t.affiliation == AFFILIATION["HOSTILE"]),
            "unknown": sum(1 for t in active if t.affiliation == AFFILIATION["UNKNOWN"]),
            "civilian": sum(1 for t in active if t.affiliation == AFFILIATION["CIVILIAN"]),
        }

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute distance between two lat/lon points in meters."""
        R = 6371000.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    @staticmethod
    def _determine_affiliation(label: str) -> str:
        """Determine affiliation from classification label."""
        if label in FRIENDLY_LABELS:
            return AFFILIATION["FRIENDLY"]
        elif label == "unknown":
            return AFFILIATION["UNKNOWN"]
        elif label in {"AM radio"}:
            return AFFILIATION["CIVILIAN"]
        elif label in {"Airborne-detection", "Airborne-range", "Air-Ground-MTI", "EW-Jammer"}:
            return AFFILIATION["HOSTILE"]
        else:
            return AFFILIATION["UNKNOWN"]
