"""
Geolocation Engine
Estimates emitter locations from multi-receiver RSSI and ToA observations.
Supports:
- RSSI-based trilateration (nonlinear least squares)
- TDoA-based multilateration (hyperbolic positioning)
- Hybrid RSSI + TDoA
- Kalman filtering for moving emitter tracks
- GDOP (Geometric Dilution of Precision) uncertainty estimation
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import minimize, least_squares
from scipy.linalg import lstsq

logger = logging.getLogger(__name__)

# Speed of light (m/s)
C = 2.998e8

# Earth radius for lat/lon conversions (m)
EARTH_RADIUS = 6371000.0


@dataclass
class ReceiverInfo:
    receiver_id: str
    latitude: float
    longitude: float
    sensitivity_dbm: float
    timing_accuracy_ns: float

    def xy(self) -> tuple[float, float]:
        """Return local XY coordinates in meters relative to first receiver."""
        return (self.longitude, self.latitude)


@dataclass
class PathLossModel:
    rssi_ref_dbm: float
    d_ref_m: float
    path_loss_exponent: float
    rssi_noise_std_db: float

    def rssi_to_distance(self, rssi_dbm: float) -> float:
        """Convert RSSI measurement to estimated distance in meters."""
        exponent = (self.rssi_ref_dbm - rssi_dbm) / (10.0 * self.path_loss_exponent)
        return self.d_ref_m * (10.0 ** exponent)

    def distance_to_rssi(self, distance_m: float) -> float:
        """Convert distance to expected RSSI."""
        return self.rssi_ref_dbm - 10.0 * self.path_loss_exponent * np.log10(distance_m / self.d_ref_m)


@dataclass
class GeoResult:
    latitude: float
    longitude: float
    uncertainty_m: float          # Estimated position error radius (meters)
    method: str                   # "rssi", "tdoa", "hybrid", "single"
    n_receivers: int
    gdop: float = 0.0             # Geometric dilution of precision
    residual: float = 0.0         # Optimization residual


def latlon_to_xy(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Convert lat/lon to local x/y in meters."""
    x = (lon - ref_lon) * np.pi / 180.0 * EARTH_RADIUS * np.cos(ref_lat * np.pi / 180.0)
    y = (lat - ref_lat) * np.pi / 180.0 * EARTH_RADIUS
    return x, y


def xy_to_latlon(x: float, y: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Convert local x/y in meters back to lat/lon."""
    lat = ref_lat + y / EARTH_RADIUS * 180.0 / np.pi
    lon = ref_lon + x / (EARTH_RADIUS * np.cos(ref_lat * np.pi / 180.0)) * 180.0 / np.pi
    return lat, lon


class GeolocatorEngine:
    """
    Emitter geolocation from multi-receiver observations.
    """

    def __init__(self, receivers: list[ReceiverInfo], path_loss: PathLossModel):
        self.receivers: dict[str, ReceiverInfo] = {r.receiver_id: r for r in receivers}
        self.path_loss = path_loss

        if receivers:
            self._ref_lat = receivers[0].latitude
            self._ref_lon = receivers[0].longitude
        else:
            self._ref_lat = 49.26
            self._ref_lon = -123.25

        # Pre-compute receiver XY positions
        self._rx_xy: dict[str, tuple[float, float]] = {}
        for rx in receivers:
            self._rx_xy[rx.receiver_id] = latlon_to_xy(
                rx.latitude, rx.longitude,
                self._ref_lat, self._ref_lon
            )

    def geolocate(
        self,
        observations: list[dict],
    ) -> Optional[GeoResult]:
        """
        Geolocate an emitter from a list of associated observations.
        Each observation dict must have: receiver_id, rssi_dbm, time_of_arrival_ns (optional)
        """
        if not observations:
            return None

        # Filter to observations with known receivers
        valid_obs = [o for o in observations if o.get("receiver_id") in self.receivers]
        if not valid_obs:
            return None

        # Try TDoA if we have enough receivers with valid ToA
        toa_obs = [o for o in valid_obs if o.get("time_of_arrival_ns") is not None]

        if len(toa_obs) >= 3:
            result = self._geolocate_hybrid(valid_obs, toa_obs)
        elif len(valid_obs) >= 3:
            result = self._geolocate_rssi(valid_obs)
        elif len(valid_obs) == 2:
            result = self._geolocate_rssi_2rx(valid_obs)
        elif len(valid_obs) == 1:
            result = self._geolocate_single(valid_obs[0])
        else:
            return None

        return result

    def _geolocate_rssi(self, obs: list[dict]) -> GeoResult:
        """RSSI-based trilateration using nonlinear least squares."""

        rx_positions = []
        distances = []
        weights = []

        for o in obs:
            rx_id = o["receiver_id"]
            if rx_id not in self._rx_xy:
                continue
            xy = self._rx_xy[rx_id]
            d = self.path_loss.rssi_to_distance(o["rssi_dbm"])
            # Weight by inverse variance (closer to receiver noise floor = less reliable)
            rx = self.receivers[rx_id]
            snr_weight = max(0.1, 1.0 + o.get("snr_estimate_db", 0.0) / 20.0)
            rx_positions.append(xy)
            distances.append(d)
            weights.append(snr_weight)

        if len(rx_positions) < 2:
            return None

        rx_arr = np.array(rx_positions)
        d_arr = np.array(distances)
        w_arr = np.array(weights)

        # Initial estimate: weighted centroid of receiver positions
        x0 = np.average(rx_arr[:, 0], weights=w_arr)
        y0 = np.average(rx_arr[:, 1], weights=w_arr)

        def residuals(pos):
            diffs = []
            for i, (rx, d_meas) in enumerate(zip(rx_positions, distances)):
                d_est = np.sqrt((pos[0] - rx[0])**2 + (pos[1] - rx[1])**2)
                diffs.append(w_arr[i] * (d_est - d_meas))
            return diffs

        result = least_squares(residuals, [x0, y0], method="lm", max_nfev=1000)
        x_est, y_est = result.x

        # Estimate uncertainty from residual
        residual_rms = np.sqrt(np.mean(np.array(residuals(result.x))**2))
        uncertainty = residual_rms / np.mean(w_arr)

        # GDOP estimation
        gdop = self._compute_gdop(result.x, rx_arr)

        lat, lon = xy_to_latlon(x_est, y_est, self._ref_lat, self._ref_lon)

        return GeoResult(
            latitude=round(lat, 6),
            longitude=round(lon, 6),
            uncertainty_m=round(min(uncertainty, 5000.0), 1),
            method="rssi",
            n_receivers=len(rx_positions),
            gdop=round(gdop, 2),
            residual=round(float(result.cost), 2),
        )

    def _geolocate_rssi_2rx(self, obs: list[dict]) -> GeoResult:
        """2-receiver RSSI: constrained estimate along line of intersection."""
        # With only 2 circles, there are 2 intersection points; pick midpoint
        result = self._geolocate_rssi(obs)
        if result:
            result.uncertainty_m = max(result.uncertainty_m, 500.0)
            result.method = "rssi_2rx"
        return result

    def _geolocate_tdoa(self, toa_obs: list[dict], x0: float = None, y0: float = None) -> Optional[GeoResult]:
        """TDoA-based multilateration using hyperbolic equations."""
        if len(toa_obs) < 3:
            return None

        # Anchor selection: Use highest-SNR/RSSI observation as reference for better stability
        ref = max(toa_obs, key=lambda o: o.get("rssi_dbm", -999))
        ref_xy = self._rx_xy[ref["receiver_id"]]
        
        # Other observations (all except the reference)
        other_obs = [o for o in toa_obs if o["receiver_id"] != ref["receiver_id"]]

        # Build system of hyperbolic equations
        # For each pair (ref, rx_i): (d_i - d_ref) = c * (t_i - t_ref)
        A_rows = []
        b_rows = []

        for o in other_obs:
            rx_id = o["receiver_id"]
            if rx_id not in self._rx_xy:
                continue
            rx_xy = self._rx_xy[rx_id]
            tdoa_ns = o["time_of_arrival_ns"] - ref["time_of_arrival_ns"]
            range_diff = tdoa_ns * 1e-9 * C  # Convert ns to meters

            # Linearized TDOA equation
            x1, y1 = ref_xy
            x2, y2 = rx_xy
            A_rows.append([2*(x2-x1), 2*(y2-y1)])
            b_rows.append(range_diff**2 - x2**2 + x1**2 - y2**2 + y1**2)

        if len(A_rows) < 2:
            return None

        A = np.array(A_rows)
        b = np.array(b_rows)

        try:
            if x0 is not None and y0 is not None:
                pos = np.array([x0, y0])
            else:
                pos, residuals, _, _ = lstsq(A, b)
        except Exception:
            return None

        # Refine with nonlinear optimization
        def tdoa_residuals(p):
            diffs = []
            d_ref = np.sqrt((p[0] - ref_xy[0])**2 + (p[1] - ref_xy[1])**2)
            for o in other_obs:
                rx_id = o["receiver_id"]
                if rx_id not in self._rx_xy:
                    continue
                rx_xy = self._rx_xy[rx_id]
                d_i = np.sqrt((p[0] - rx_xy[0])**2 + (p[1] - rx_xy[1])**2)
                tdoa_ns = o["time_of_arrival_ns"] - ref["time_of_arrival_ns"]
                range_diff_meas = tdoa_ns * 1e-9 * C
                diffs.append((d_i - d_ref) - range_diff_meas)
            return diffs

        refined = least_squares(tdoa_residuals, pos, max_nfev=2000)
        x_est, y_est = refined.x[:2]

        lat, lon = xy_to_latlon(x_est, y_est, self._ref_lat, self._ref_lon)
        residual_rms = np.sqrt(np.mean(np.array(tdoa_residuals(refined.x))**2)) if len(tdoa_residuals(refined.x)) > 0 else 999

        return GeoResult(
            latitude=round(lat, 6),
            longitude=round(lon, 6),
            uncertainty_m=round(min(residual_rms * 2, 2000.0), 1),
            method="tdoa",
            n_receivers=len(toa_obs),
            gdop=self._compute_gdop(refined.x, np.array([self._rx_xy[o["receiver_id"]] for o in toa_obs if o["receiver_id"] in self._rx_xy])),
            residual=round(float(refined.cost), 2),
        )

    def _geolocate_hybrid(self, rssi_obs: list[dict], toa_obs: list[dict]) -> GeoResult:
        """
        Hybrid RSSI + TDoA: use TDoA for primary location, RSSI for validation/refinement.
        """
        rssi_result = self._geolocate_rssi(rssi_obs)
        
        # Use RSSI estimate as initial seed for TDoA to avoid local minima
        x0, y0 = None, None
        if rssi_result:
            x0, y0 = latlon_to_xy(rssi_result.latitude, rssi_result.longitude, self._ref_lat, self._ref_lon)
            
        tdoa_result = self._geolocate_tdoa(toa_obs, x0=x0, y0=y0)

        if tdoa_result is None:
            return rssi_result
        if rssi_result is None:
            return tdoa_result

        # Inverse-variance weighting (Dynamic fusion)
        # We favor the method with lower uncertainty
        tdoa_var = max(tdoa_result.uncertainty_m, 5.0) ** 2
        rssi_var = max(rssi_result.uncertainty_m, 5.0) ** 2
        
        inv_sum = (1.0 / tdoa_var) + (1.0 / rssi_var)
        tdoa_weight = (1.0 / tdoa_var) / inv_sum
        rssi_weight = (1.0 / rssi_var) / inv_sum

        tdoa_x, tdoa_y = latlon_to_xy(tdoa_result.latitude, tdoa_result.longitude, self._ref_lat, self._ref_lon)
        rssi_x, rssi_y = latlon_to_xy(rssi_result.latitude, rssi_result.longitude, self._ref_lat, self._ref_lon)

        x_fused = tdoa_weight * tdoa_x + rssi_weight * rssi_x
        y_fused = tdoa_weight * tdoa_y + rssi_weight * rssi_y

        lat, lon = xy_to_latlon(x_fused, y_fused, self._ref_lat, self._ref_lon)
        uncertainty = tdoa_weight * tdoa_result.uncertainty_m + rssi_weight * rssi_result.uncertainty_m

        return GeoResult(
            latitude=round(lat, 6),
            longitude=round(lon, 6),
            uncertainty_m=round(uncertainty, 1),
            method="hybrid",
            n_receivers=max(rssi_result.n_receivers, tdoa_result.n_receivers),
            gdop=min(tdoa_result.gdop, rssi_result.gdop),
            residual=tdoa_result.residual,
        )

    def _geolocate_single(self, obs: dict) -> GeoResult:
        """Single receiver: estimate position toward receiver direction."""
        rx = self.receivers[obs["receiver_id"]]
        d = self.path_loss.rssi_to_distance(obs["rssi_dbm"])

        # We only know distance, not direction - place emitter at receiver position
        # with uncertainty = estimated distance
        return GeoResult(
            latitude=round(rx.latitude, 6),
            longitude=round(rx.longitude, 6),
            uncertainty_m=round(min(d, 5000.0), 1),
            method="single",
            n_receivers=1,
            gdop=99.0,
        )

    def _compute_gdop(self, pos: np.ndarray, rx_positions: np.ndarray) -> float:
        """Compute Geometric Dilution of Precision."""
        if len(rx_positions) < 2:
            return 10.0
        try:
            H = []
            for rx in rx_positions:
                d = np.sqrt((pos[0] - rx[0])**2 + (pos[1] - rx[1])**2) + 1e-10
                H.append([(pos[0] - rx[0]) / d, (pos[1] - rx[1]) / d])
            H = np.array(H)
            HTH = H.T @ H
            HTH += np.eye(2) * 0.001  # Regularization
            Q = np.linalg.inv(HTH)
            trace = np.trace(Q)
            if trace <= 0: return 10.0
            gdop = float(np.sqrt(trace))
            return min(gdop, 50.0)
        except Exception:
            return 10.0



class KalmanTracker:
    """
    Simple 2D Kalman filter for tracking moving emitters.
    State: [x, y, vx, vy] (position + velocity in local coords)
    """

    def __init__(self, init_x: float, init_y: float, init_uncertainty: float = 500.0):
        # State: [x, y, vx, vy]
        self.x = np.array([init_x, init_y, 0.0, 0.0], dtype=float)

        # Covariance matrix (uncertainty)
        self.P = np.diag([
            init_uncertainty**2,
            init_uncertainty**2,
            1000.0**2,  # velocity uncertainty
            1000.0**2,
        ])

        # Process noise (emitter can accelerate)
        # Position noise is small to keep track stable, velocity noise allows for movement
        self.Q = np.diag([5.0**2, 5.0**2, 15.0**2, 15.0**2])


        # Measurement noise (position uncertainty from geolocation)
        self.R_base = np.diag([200.0**2, 200.0**2])

        # Transition matrix (constant velocity model)
        self.F = np.eye(4)

        # Measurement matrix (observe position only)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def predict(self, dt: float = 1.0):
        """Predict next state."""
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, x_meas: float, y_meas: float, uncertainty_m: float = 200.0):
        """Update with new position measurement."""
        R = np.diag([uncertainty_m**2, uncertainty_m**2])
        z = np.array([x_meas, y_meas])
        y_innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y_innov
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def position_uncertainty(self) -> float:
        return float(np.sqrt((self.P[0, 0] + self.P[1, 1]) / 2))
