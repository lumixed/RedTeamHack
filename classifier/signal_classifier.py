"""
Signal Classifier Module
Trains on labeled IQ waveform data and classifies signals.
Supports both known friendly signal classification and
out-of-distribution (hostile/civilian) anomaly detection.
"""

import numpy as np
import joblib
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score
from sklearn.covariance import EllipticEnvelope
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ─── Signal catalog ───────────────────────────────────────────────────────────
FRIENDLY_LABELS = {
    "Radar-Altimeter",
    "Satcom",
    "short-range",
}

# Note: The server uses these exact labels (case-insensitive)
HOSTILE_LABELS = {
    "Airborne-detection",
    "Airborne-range",
    "Air-Ground-MTI",
    "EW-Jammer",
}

CIVILIAN_LABELS = {
    "AM radio",
}

ALL_KNOWN_LABELS = FRIENDLY_LABELS | HOSTILE_LABELS | CIVILIAN_LABELS

MODEL_DIR = Path(__file__).parent.parent / "models"

OOD_NU = 0.05
OOD_PERCENTILE = 100 * OOD_NU


def extract_features(iq_snapshot: list) -> np.ndarray:
    """
    Extract rich feature vector from 256-element IQ snapshot.
    Elements 0-127: I components
    Elements 128-255: Q components
    Sample rate: 10 MS/s
    """
    iq = np.array(iq_snapshot, dtype=np.float32)
    if len(iq) != 256:
        iq = np.pad(iq, (0, max(0, 256 - len(iq))))[:256]

    I = iq[:128]
    Q = iq[128:]

    # Complex representation
    z = I + 1j * Q

    # ── Amplitude/envelope features ─────────────────────────────────
    amplitude = np.abs(z)
    amp_mean = np.mean(amplitude)
    amp_std = np.std(amplitude)
    amp_max = np.max(amplitude)
    amp_min = np.min(amplitude)
    amp_range = amp_max - amp_min
    amp_skew = _skewness(amplitude)
    amp_kurt = _kurtosis(amplitude)
    crest_factor = amp_max / (amp_mean + 1e-10)

    # ── High-Resolution Spectral features (512-PT FFT) ────────────────
    spectrum_hires = np.abs(np.fft.fft(z, n=512)) ** 2
    spectrum_hires_norm = spectrum_hires / (np.sum(spectrum_hires) + 1e-10)
    peak_freq_hires = float(np.argmax(spectrum_hires_norm[:256]))
    
    # ── Haar Wavelet Decomposition (Simple Filter Bank) ──────────────
    # Level 1 decomposition
    cA = (z[0::2] + z[1::2]) / np.sqrt(2) # Low pass
    cD = (z[0::2] - z[1::2]) / np.sqrt(2) # High pass
    energy_low_wavelet = float(np.sum(np.abs(cA)**2))
    energy_high_wavelet = float(np.sum(np.abs(cD)**2))

    # ── Phase features ───────────────────────────────────────────────
    phase = np.angle(z)
    phase_diff = np.diff(np.unwrap(phase))
    phase_mean = np.mean(phase)
    phase_std = np.std(phase)
    phase_diff_std = np.std(phase_diff)
    phase_diff_mean = np.mean(phase_diff)

    # ── Instantaneous frequency ──────────────────────────────────────
    inst_freq = phase_diff / (2 * np.pi * 1e-7)  # Hz (dt = 1/10MHz = 100ns)
    freq_mean = np.mean(inst_freq)
    freq_std = np.std(inst_freq)
    freq_range = np.ptp(inst_freq)

    # ── Power / energy ───────────────────────────────────────────────
    power = amplitude ** 2
    total_power = np.sum(power)
    i_power = np.sum(I ** 2)
    q_power = np.sum(Q ** 2)
    power_ratio = i_power / (q_power + 1e-10)

    # ── Spectral features (FFT) ──────────────────────────────────────
    spectrum = np.abs(np.fft.fft(z)) ** 2
    spectrum_norm = spectrum / (np.sum(spectrum) + 1e-10)
    freqs = np.fft.fftfreq(128, d=1e-7)

    spec_mean = np.mean(spectrum_norm)
    spec_std = np.std(spectrum_norm)
    spec_entropy = -np.sum(spectrum_norm * np.log2(spectrum_norm + 1e-10))
    peak_freq_idx = np.argmax(spectrum_norm[:64])
    spectral_centroid = np.sum(np.arange(len(spectrum_norm)) * spectrum_norm) / (np.sum(spectrum_norm) + 1e-10)
    spectral_flatness = np.exp(np.mean(np.log(spectrum + 1e-10))) / (np.mean(spectrum) + 1e-10)

    # Top 5 spectral peaks
    top5_peaks = np.sort(spectrum_norm[:64])[-5:][::-1]

    # ── Noise floor estimate ─────────────────────────────────────────
    sorted_amp = np.sort(amplitude)
    noise_floor = np.mean(sorted_amp[:16])  # Bottom 12.5% = noise
    peak_to_noise = amp_max / (noise_floor + 1e-10)

    # ── Pulsed signal features (Low-SNR Robust) ──────────────────────
    # Use a threshold that adapts to the noise floor rather than the mean
    if peak_to_noise > 3.0:
        # Strong signal: threshold halfway between noise and peak
        pulse_threshold = noise_floor + (amp_max - noise_floor) * 0.3
    else:
        # Weak signal: tighter threshold near noise floor
        pulse_threshold = noise_floor * 1.5

    above = amplitude > pulse_threshold
    duty_cycle = np.mean(above)

    # Zero crossings of amplitude envelope (proxy for pulse transitions)
    amp_centered = amplitude - pulse_threshold
    zcr_amp = np.sum(np.diff(np.sign(amp_centered)) != 0) / len(amplitude)

    # ── BPSK detection (phase transitions of ~180°) ──────────────────
    phase_jumps_180 = np.sum(np.abs(phase_diff) > np.pi * 0.7) / len(phase_diff)

    # ── FMCW detection (linear frequency sweep) ──────────────────────
    freq_linearity = np.corrcoef(np.arange(len(inst_freq)), inst_freq)[0, 1] if len(inst_freq) > 1 else 0.0

    # ── ASK detection (amplitude on/off pattern) ─────────────────────
    ask_ratio = amp_std / (amp_mean + 1e-10)

    # ── Advanced Features for Satcom/Radar distinction ───────────────
    # Spectral Rolloff (85% of total power)
    cum_spec = np.cumsum(spectrum_norm)
    spec_rolloff = float(np.searchsorted(cum_spec, 0.85)) / len(spectrum_norm)
    
    # Peak-to-Average Power Ratio (PAPR)
    papr = float(amp_max**2 / (total_power + 1e-10))

    # Spectral Shape Stats
    spec_skew = _skewness(spectrum_norm)
    spec_kurt = _kurtosis(spectrum_norm)
    
    # Band Energy (sub-band distribution)
    # Total spectrum length is 128
    energy_low = np.sum(spectrum_norm[:32])
    energy_mid_low = np.sum(spectrum_norm[32:64])
    energy_mid_high = np.sum(spectrum_norm[64:96])
    energy_high = np.sum(spectrum_norm[96:])

    # ── Higher Order Cumulants (HOC) ─────────────────────────────────
    # Normalized moments capture modulation "envelope" shape
    z_norm = (z - np.mean(z)) / (np.std(z) + 1e-10)
    e_z2 = np.mean(z_norm**2)
    e_absz2 = np.mean(np.abs(z_norm)**2)
    e_z4 = np.mean(z_norm**4)
    e_absz4 = np.mean(np.abs(z_norm)**4)
    
    c40 = e_z4 - 3 * (e_z2**2)
    c42 = e_absz4 - np.abs(e_z2)**2 - 2 * (e_absz2**2)
    
    c40_norm = np.abs(c40)
    c42_norm = np.abs(c42)

    # ── Phase Histogram (Catch M-PSK signatures) ─────────────────────
    # Standardize phase to [0, 2pi]
    phase_std_range = (phase + np.pi) % (2 * np.pi)
    phase_hist, _ = np.histogram(phase_std_range, bins=8, range=(0, 2*np.pi), density=True)

    # ── Autocorrelation (Multi-Lag) ──────────────────────────────────
    def get_autocorr(lag):
        if len(z) > lag:
            r = np.sum(z[lag:] * np.conj(z[:-lag])) / (np.sum(np.abs(z)**2) + 1e-10)
            return np.abs(r), np.angle(r)
        return 0.0, 0.0

    r1_mag, r1_phase = get_autocorr(1)
    r2_mag, r2_phase = get_autocorr(2)
    r4_mag, r4_phase = get_autocorr(4)
    r8_mag, r8_phase = get_autocorr(8)
    
    # Autocorrelation Peak (Symbol Rate Proxy)
    # Exclude the DC peak at lag 0
    all_lags = [get_autocorr(l)[0] for l in range(1, 32)]
    r_peak_val = np.max(all_lags)
    r_peak_lag = np.argmax(all_lags) + 1

    # ── Block-based Temporal Features (Intra-pulse dynamics) ──────────
    # Divide 128 samples into 4 blocks of 32
    z_blocks = np.split(z, 4)
    block_amps = [np.std(np.abs(b)) for b in z_blocks]
    block_phases = [np.std(np.unwrap(np.angle(b))) for b in z_blocks]
    block_amp_var = np.var(block_amps)
    block_phase_var = np.var(block_phases)

    # ── Fractional Moments ───────────────────────────────────────────
    amp_05 = np.mean(np.sqrt(amplitude + 1e-10))
    amp_15 = np.mean(amplitude**1.5)

    # ── LPC (Linear Predictive Coding) Coefficients (8) ──────────────
    # Using Yule-Walker method with autocorrelation
    def get_lpc(p=8):
        # We already have autocorrelation up to lag 31 from previous step
        # Let's use it for the Yule-Walker equations
        r_lpc = [1.0] + all_lags[:p]
        # Construct Toeplitz matrix
        from scipy.linalg import toeplitz, solve
        R = toeplitz(r_lpc[:p])
        r_vec = np.array(r_lpc[1:p+1])
        try:
            a = solve(R, r_vec)
            return a.tolist()
        except:
            return [0.0] * p

    try:
        from scipy.linalg import toeplitz, solve
        lpc_coeffs = get_lpc(8)
    except:
        lpc_coeffs = [0.0] * 8

    # ── Phase Stability ──────────────────────────────────────────────
    # Variance of the phase difference (jitter)
    phase_jitter = np.var(np.abs(phase_diff))

    # ── Higher-order statistics ──────────────────────────────────────
    i_std = np.std(I)
    q_std = np.std(Q)
    iq_corr = np.corrcoef(I, Q)[0, 1] if i_std > 0 and q_std > 0 else 0.0

    # ── Zero-crossing rate of raw I and Q ────────────────────────────
    zcr_i = np.sum(np.diff(np.sign(I)) != 0) / len(I)
    zcr_q = np.sum(np.diff(np.sign(Q)) != 0) / len(Q)

    features = np.array([
        # Amplitude stats (8)
        amp_mean, amp_std, amp_max, amp_min, amp_range,
        amp_skew, amp_kurt, crest_factor,
        # Phase stats (5)
        phase_mean, phase_std,
        phase_diff_std, phase_diff_mean, phase_jumps_180,
        # Frequency stats (4)
        freq_mean, freq_std, freq_range, freq_linearity,
        # Power stats (4)
        total_power, i_power, q_power, power_ratio,
        # Spectral stats (6)
        spec_mean, spec_std, spec_entropy,
        spectral_centroid, spectral_flatness, peak_freq_idx,
        # Top 5 spectral peaks (5)
        *top5_peaks,
        # Advanced modulation stats (8)
        spec_rolloff, papr, spec_skew, spec_kurt,
        energy_low, energy_mid_low, energy_mid_high, energy_high,
        # HOC (2)
        c40_norm, c42_norm,
        # Phase Histogram (8)
        *phase_hist,
        # High-res and Wavelet (3)
        peak_freq_hires, energy_low_wavelet, energy_high_wavelet,
        # Autocorrelation Peak (2)
        r_peak_val, float(r_peak_lag),
        # Block-based temporal (10)
        *block_amps, *block_phases, block_amp_var, block_phase_var,
        # Fractional moments (2)
        amp_05, amp_15,
        # LPC (8)
        *lpc_coeffs,
        # Phase Stability (1)
        phase_jitter,
        # IQ correlation stats (4)
        i_std, q_std, iq_corr, noise_floor,
        # ZCR (2)
        zcr_i, zcr_q,
        # ASK and PAPR (2)
        ask_ratio, papr,
        # Pulsed features (2)
        duty_cycle, zcr_amp,
    ], dtype=np.float32)


    return features


def _skewness(x: np.ndarray) -> float:
    """Compute skewness of array."""
    mu = np.mean(x)
    sig = np.std(x)
    if sig == 0:
        return 0.0
    return float(np.mean(((x - mu) / sig) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Compute kurtosis of array."""
    mu = np.mean(x)
    sig = np.std(x)
    if sig == 0:
        return 0.0
    return float(np.mean(((x - mu) / sig) ** 4)) - 3.0


class DeepSignalNet(nn.Module):
    """
    1D-CNN architecture for deep feature extraction from I/Q data.
    Input: (Batch, 2, 128)
    """
    def __init__(self, num_classes=3, feature_mode=False):
        super(DeepSignalNet, self).__init__()
        self.feature_mode = feature_mode
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, 256) -> Reshape to (B, 2, 128)
        x = x.view(-1, 2, 128)
        features = self.conv_layers(x).squeeze(-1)
        if self.feature_mode:
            return features
        return self.fc(features)

class SignalClassifier:
    """
    Three-stage signal classifier:
    1. Deep CNN: extracts deep latent features from raw IQ
    2. Hybrid Ensemble: combines CNN features with manual features
    3. Anomaly detector: flags out-of-distribution (hostile/civilian) signals
    """

    def __init__(self):
        self.cnn = None
        self.friendly_classifier = None
        self.scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)
        self.label_encoder = LabelEncoder()
        self.anomaly_detector = None
        self.is_trained = False
        self._ood_threshold = -0.1
        MODEL_DIR.mkdir(exist_ok=True)

    def train(self, X_feat: np.ndarray, X_raw: np.ndarray, y: np.ndarray) -> dict:
        """
        Train on labeled friendly IQ data.
        X_feat: (N, feature_dim) manual features
        X_raw: (N, 256) raw IQ data
        y: (N,) string labels
        Returns training metrics dict.
        """
        logger.info(f"Training on {len(X_feat)} samples, {len(np.unique(y))} classes")

        # 1. Encode labels
        y_enc = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)

        # 2. Train Deep CNN Feature Extractor
        logger.info("Training Deep 1D-CNN Feature Extractor...")
        self.cnn = DeepSignalNet(num_classes=num_classes)
        
        # Prepare data for torch
        X_raw_t = torch.tensor(X_raw, dtype=torch.float32)
        y_t = torch.tensor(y_enc, dtype=torch.long)
        dataset = TensorDataset(X_raw_t, y_t)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        optimizer = optim.Adam(self.cnn.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        self.cnn.train()
        for epoch in range(10): # Quick training for hackathon context
            total_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.cnn(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"CNN Epoch {epoch+1}/10, Loss: {total_loss/len(loader):.4f}")

        # 3. Extract Deep Features
        self.cnn.eval()
        self.cnn.feature_mode = True
        with torch.no_grad():
            X_deep = self.cnn(X_raw_t).numpy()
        self.cnn.feature_mode = False # Reset for normal use if needed

        # 4. Hybridize Features (Manual + Deep)
        X_hybrid = np.hstack([X_feat, X_deep])
        logger.info(f"Hybrid feature vector dimension: {X_hybrid.shape[1]}")

        # 5. Scale and Split
        X_scaled = self.scaler.fit_transform(X_hybrid)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        # 6. Train Ensemble Classifier (HGB + MLP)
        logger.info("Training Hybrid Ensemble...")
        hgb = HistGradientBoostingClassifier(random_state=42, early_stopping=True)
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=42)
        
        voter = VotingClassifier(
            estimators=[('hgb', hgb), ('mlp', mlp)],
            voting='soft'
        )
        self.friendly_classifier = CalibratedClassifierCV(voter, method='sigmoid', cv=3)
        self.friendly_classifier.fit(X_tr, y_tr)

        # Evaluate
        y_pred = self.friendly_classifier.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="macro")
        report = classification_report(
            y_val, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        logger.info(f"Hybrid Classifier F1 (macro): {f1:.3f}")

        # 7. Train One-Class SVM anomaly detector on friendly data only
        self.anomaly_detector = OneClassSVM(kernel="rbf", nu=OOD_NU, gamma="scale")
        self.anomaly_detector.fit(X_tr)

        # Calibrate the cutoff to the same outlier fraction the detector was fitted
        # for. Setting it looser than nu overrides that fit and flags friendly
        # signals as novel, which also splits their observations across
        # association groups and forks duplicate tracks.
        scores = self.anomaly_detector.decision_function(X_val)
        self._ood_threshold = float(np.percentile(scores, OOD_PERCENTILE))
        
        self.is_trained = True
        return {
            "f1_macro": round(f1, 4),
            "n_samples": len(X_feat),
            "classes": list(self.label_encoder.classes_),
            "per_class": {k: v for k, v in report.items() if k in self.label_encoder.classes_},
        }

    def predict(self, iq_snapshot: list) -> dict:
        """
        Classify a single IQ snapshot.
        Returns dict with label, confidence, is_friendly, is_anomaly.
        """
        raw_iq = np.array(iq_snapshot, dtype=np.float32)
        if len(raw_iq) != 256:
            raw_iq = np.pad(raw_iq, (0, max(0, 256 - len(raw_iq))))[:256]
            
        manual_features = extract_features(iq_snapshot)
        return self.predict_hybrid(manual_features.reshape(1, -1), raw_iq.reshape(1, -1))[0]

    def predict_hybrid(self, X_feat: np.ndarray, X_raw: np.ndarray) -> list:
        """
        Classify using both manual and deep features.
        """
        if not self.is_trained:
            return [self._unknown_result() for _ in range(len(X_feat))]

        # 1. Extract Deep Features
        self.cnn.eval()
        self.cnn.feature_mode = True
        with torch.no_grad():
            X_raw_t = torch.tensor(X_raw, dtype=torch.float32)
            X_deep = self.cnn(X_raw_t).numpy()
        self.cnn.feature_mode = False

        # 2. Hybridize
        X_hybrid = np.hstack([X_feat, X_deep])
        X_scaled = self.scaler.transform(X_hybrid)

        # 3. Anomaly detection
        ood_scores = self.anomaly_detector.decision_function(X_scaled)
        is_anomaly = ood_scores < self._ood_threshold

        # 4. Friendly classification
        proba = self.friendly_classifier.predict_proba(X_scaled)
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = self.label_encoder.inverse_transform(pred_idx)
        confidences = proba[np.arange(len(proba)), pred_idx]

        results = []
        for i in range(len(X_feat)):
            feat = X_feat[i]
            features_dict = {
                "amp_std": float(feat[1]),
                "freq_mean": float(feat[13]),
                "freq_std": float(feat[14]),
                "total_power": float(feat[17]),
                "spectral_flatness": float(feat[25]),
                "duty_cycle": float(feat[84]),
                "ask_ratio": float(feat[82]),
                "papr": float(feat[83]),
                "freq_linearity": float(feat[16]),
                "phase_jumps_180": float(feat[12]),
                "crest_factor": float(feat[7]),
            }

            friendly_conf = float(confidences[i])
            ood_score = round(float(ood_scores[i]), 4)

            if is_anomaly[i]:
                ood_conf = float(1.0 - (ood_scores[i] - self._ood_threshold) /
                                  (abs(self._ood_threshold) + 1e-10))
                ood_conf = max(0.5, min(0.99, ood_conf))
                results.append({
                    "label": "unknown",
                    "confidence": round(ood_conf, 3),
                    "is_friendly": False,
                    "is_anomaly": True,
                    "friendly_guess": str(pred_labels[i]),
                    "friendly_confidence": round(friendly_conf, 3),
                    "ood_score": ood_score,
                    "features": features_dict,
                })
            else:
                results.append({
                    "label": str(pred_labels[i]),
                    "confidence": round(friendly_conf, 3),
                    "is_friendly": True,
                    "is_anomaly": False,
                    "friendly_guess": str(pred_labels[i]),
                    "friendly_confidence": round(friendly_conf, 3),
                    "ood_score": ood_score,
                    "features": features_dict,
                })

        return results

    def _unknown_result(self) -> dict:
        return {
            "label": "unknown",
            "confidence": 0.5,
            "is_friendly": False,
            "is_anomaly": True,
            "friendly_guess": None,
            "friendly_confidence": 0.0,
            "ood_score": 0.0,
        }

    def save(self, path: str = None):
        """Save model to disk."""
        if path is None:
            path = str(MODEL_DIR / "classifier.joblib")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # We need to handle the torch model separately or within the dict
        cnn_state = self.cnn.state_dict() if self.cnn else None
        
        joblib.dump({
            "cnn_state": cnn_state,
            "cnn_num_classes": len(self.label_encoder.classes_) if self.is_trained else 0,
            "friendly_classifier": self.friendly_classifier,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "anomaly_detector": self.anomaly_detector,
            "ood_threshold": self._ood_threshold,
            "is_trained": self.is_trained,
        }, path)
        logger.info(f"Model (Hybrid) saved to {path}")

    def load(self, path: str = None) -> bool:
        """Load model from disk. Returns True if successful."""
        if path is None:
            path = str(MODEL_DIR / "classifier.joblib")
        if not os.path.exists(path):
            logger.warning(f"No saved model found at {path}")
            return False
            
        data = joblib.load(path)
        self.is_trained = data["is_trained"]
        self.label_encoder = data["label_encoder"]
        
        if self.is_trained and data.get("cnn_state"):
            num_classes = data.get("cnn_num_classes", len(self.label_encoder.classes_))
            self.cnn = DeepSignalNet(num_classes=num_classes)
            self.cnn.load_state_dict(data["cnn_state"])
            self.cnn.eval()
            
        self.friendly_classifier = data["friendly_classifier"]
        self.scaler = data["scaler"]
        self.anomaly_detector = data["anomaly_detector"]
        self._ood_threshold = data["ood_threshold"]
        
        logger.info(f"Model (Hybrid) loaded from {path}")
        return True


def load_training_data(hdf5_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load labeled IQ training data from HDF5 file.
    Returns (X_features, X_raw, y_labels) arrays.
    """
    import h5py
    import ast
    logger.info(f"Loading training data from {hdf5_path}")

    X_raw_list = []
    y_list = []

    with h5py.File(hdf5_path, "r") as f:
        for key in f.keys():
            try:
                t = ast.literal_eval(key)
                if not isinstance(t, tuple) or len(t) < 2:
                    continue
                label = str(t[1])
            except (ValueError, SyntaxError):
                continue

            dataset = f[key]
            data = dataset[()]

            if data.ndim == 2 and data.shape[1] == 256:
                for sample in data:
                    X_raw_list.append(sample)
                    y_list.append(label)
            elif data.ndim == 1 and len(data) == 256:
                X_raw_list.append(data)
                y_list.append(label)

    if not X_raw_list:
        raise ValueError("No data found in HDF5 file")

    X_raw = np.array(X_raw_list, dtype=np.float32)
    y_raw = np.array(y_list, dtype=str)

    # Extract manual features in parallel
    from joblib import Parallel, delayed
    logger.info(f"Extracting features from {len(X_raw)} samples in parallel...")
    X_feat = np.array(Parallel(n_jobs=-1)(delayed(extract_features)(x) for x in X_raw))

    logger.info(f"Loaded {len(X_feat)} samples, shape={X_feat.shape}")
    return X_feat, X_raw, y_raw
