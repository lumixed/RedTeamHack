"""
The simulated RF environment.

Holds a receiver network over Vancouver and a set of moving emitters, and
turns each emission into per-receiver observations. RSSI comes from the
path-loss model and time of arrival from true propagation delay, so both
trilateration and multilateration resolve back to the position the emitter
actually occupies.
"""

import uuid
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pipeline.geolocator import latlon_to_xy, xy_to_latlon, C
from .signals import SIGNAL_TYPES, FRIENDLY_TYPES, build, observe

RECEIVERS = [
    {"receiver_id": "RX-UBC", "latitude": 49.2606, "longitude": -123.2460},
    {"receiver_id": "RX-DOWNTOWN", "latitude": 49.2827, "longitude": -123.1207},
    {"receiver_id": "RX-YVR", "latitude": 49.1967, "longitude": -123.1815},
    {"receiver_id": "RX-NORTHSHORE", "latitude": 49.3200, "longitude": -123.0724},
    {"receiver_id": "RX-BURNABY", "latitude": 49.2488, "longitude": -122.9805},
]

PATH_LOSS = {
    "rssi_ref_dbm": -50.0,
    "d_ref_m": 1000.0,
    "path_loss_exponent": 2.8,
    "rssi_noise_std_db": 2.5,
}

SENSITIVITY_DBM = -88.0
TIMING_ACCURACY_NS = 40.0
NOISE_FLOOR_DBM = -100.0
IQ_SNR_RANGE = (10.0, 26.0)

REF_LAT = RECEIVERS[0]["latitude"]
REF_LON = RECEIVERS[0]["longitude"]

# Emitters stay inside the receiver polygon, where the geometry is strong enough
# for a well-conditioned fix. Metres east and north of the first receiver.
AREA_X = (1000.0, 18000.0)
AREA_Y = (-5500.0, 5500.0)


@dataclass
class Emitter:
    emitter_id: str
    signal_type: str
    latitude: float
    longitude: float
    heading_deg: float
    speed_mps: float
    period_s: float
    next_emit_s: float = 0.0
    waveform: np.ndarray = field(default=None, repr=False)

    def advance(self, dt, rng):
        """Move along the current heading, turning gently and bouncing at the edges."""
        self.heading_deg += rng.normal(0.0, 4.0) * dt
        x, y = latlon_to_xy(self.latitude, self.longitude, REF_LAT, REF_LON)
        rad = np.radians(self.heading_deg)
        x += np.sin(rad) * self.speed_mps * dt
        y += np.cos(rad) * self.speed_mps * dt

        if not AREA_X[0] <= x <= AREA_X[1]:
            self.heading_deg = -self.heading_deg
            x = float(np.clip(x, *AREA_X))
        if not AREA_Y[0] <= y <= AREA_Y[1]:
            self.heading_deg = 180.0 - self.heading_deg
            y = float(np.clip(y, *AREA_Y))

        self.latitude, self.longitude = xy_to_latlon(x, y, REF_LAT, REF_LON)


def rssi_at(distance_m):
    ref = PATH_LOSS["rssi_ref_dbm"]
    n = PATH_LOSS["path_loss_exponent"]
    d_ref = PATH_LOSS["d_ref_m"]
    return ref - 10.0 * n * np.log10(max(distance_m, 1.0) / d_ref)


class Scenario:
    """A running RF picture: receivers, emitters, and the observations they produce."""

    def __init__(self, n_emitters=8, seed=None):
        self.rng = np.random.default_rng(seed)
        self.clock = 0.0
        self.receivers = [dict(r) for r in RECEIVERS]
        self._rx_xy = {
            r["receiver_id"]: latlon_to_xy(r["latitude"], r["longitude"], REF_LAT, REF_LON)
            for r in self.receivers
        }
        self.emitters = [self._spawn(i) for i in range(n_emitters)]

    def _spawn(self, index):
        signal_type = SIGNAL_TYPES[index % len(SIGNAL_TYPES)]
        x = self.rng.uniform(*AREA_X)
        y = self.rng.uniform(*AREA_Y)
        lat, lon = xy_to_latlon(x, y, REF_LAT, REF_LON)
        airborne = signal_type in ("Radar-Altimeter", "Airborne-detection",
                                   "Airborne-range", "Air-Ground-MTI")
        return Emitter(
            emitter_id=f"EM-{index + 1:02d}",
            signal_type=signal_type,
            latitude=lat,
            longitude=lon,
            heading_deg=float(self.rng.uniform(0, 360)),
            speed_mps=float(self.rng.uniform(70, 140) if airborne else self.rng.uniform(0, 20)),
            period_s=float(self.rng.uniform(3.5, 4.5)),
            next_emit_s=float(self.rng.uniform(0.0, 3.0)),
        )

    def advance(self, dt):
        self.clock += dt
        for emitter in self.emitters:
            emitter.advance(dt, self.rng)

    def due(self):
        """Emitters whose next transmission is due at the current clock."""
        ready = [e for e in self.emitters if self.clock >= e.next_emit_s]
        for emitter in ready:
            emitter.next_emit_s = self.clock + emitter.period_s * self.rng.uniform(0.85, 1.15)
        return ready

    def observations_for(self, emitter, timestamp=None):
        """
        Produce one observation per receiver that can actually hear this emission.

        All receivers share the emitter's waveform and differ only in noise,
        RSSI and propagation delay.
        """
        waveform = build(emitter.signal_type, self.rng)
        emitter_xy = latlon_to_xy(emitter.latitude, emitter.longitude, REF_LAT, REF_LON)
        stamp = timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        emission_id = uuid.uuid4().hex[:8]

        out = []
        for receiver in self.receivers:
            rx_xy = self._rx_xy[receiver["receiver_id"]]
            distance = float(np.hypot(emitter_xy[0] - rx_xy[0], emitter_xy[1] - rx_xy[1]))

            rssi = rssi_at(distance) + self.rng.normal(0.0, PATH_LOSS["rssi_noise_std_db"])
            if rssi < SENSITIVITY_DBM:
                continue

            snr = rssi - NOISE_FLOOR_DBM
            iq_snr = float(np.clip(snr, *IQ_SNR_RANGE))
            toa_ns = distance / C * 1e9 + self.rng.normal(0.0, TIMING_ACCURACY_NS)

            out.append({
                "observation_id": f"{emission_id}-{receiver['receiver_id']}",
                "receiver_id": receiver["receiver_id"],
                "timestamp": stamp,
                "rssi_dbm": round(float(rssi), 2),
                "snr_estimate_db": round(float(snr), 2),
                "time_of_arrival_ns": round(float(toa_ns), 1),
                "iq_snapshot": observe(waveform, iq_snr, self.rng).tolist(),
            })

        return out

    def truth(self):
        return [
            {
                "emitter_id": e.emitter_id,
                "signal_type": e.signal_type,
                "latitude": round(e.latitude, 6),
                "longitude": round(e.longitude, 6),
                "is_friendly": e.signal_type in FRIENDLY_TYPES,
            }
            for e in self.emitters
        ]
