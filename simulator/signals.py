"""
Synthetic IQ waveform generation.

Produces 256-element snapshots in the layout the classifier expects:
elements 0-127 are I, elements 128-255 are Q, sampled at 10 MS/s.

Each emitter type is built from its actual modulation so that the
envelope and spectral statistics the pipeline keys on - crest factor,
spectral flatness, duty cycle - fall out of the waveform rather than
being painted on afterwards.
"""

import numpy as np

SAMPLE_RATE = 1e7
N_COMPLEX = 128
NYQUIST = SAMPLE_RATE / 2.0

FRIENDLY_TYPES = ["Radar-Altimeter", "Satcom", "short-range"]
HOSTILE_TYPES = ["Airborne-detection", "Airborne-range", "Air-Ground-MTI", "EW-Jammer"]
CIVILIAN_TYPES = ["AM radio"]
SIGNAL_TYPES = FRIENDLY_TYPES + HOSTILE_TYPES + CIVILIAN_TYPES

MODULATION_OF = {
    "Radar-Altimeter": "fmcw",
    "Satcom": "qpsk",
    "short-range": "ask",
    "Airborne-detection": "pulsed",
    "Airborne-range": "poly-phase",
    "Air-Ground-MTI": "pulsed-doppler",
    "EW-Jammer": "barrage-noise",
    "AM radio": "am-dsb",
}


def _time_axis():
    return np.arange(N_COMPLEX) / SAMPLE_RATE


def _pulse_envelope(start, width, rise=2):
    """Rectangular pulse with short cosine edges to avoid spectral splatter."""
    env = np.zeros(N_COMPLEX)
    stop = min(start + width, N_COMPLEX)
    env[start:stop] = 1.0
    if rise > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, rise + 2)[1:-1]))
        head = min(rise, stop - start)
        env[start:start + head] *= ramp[:head]
        tail_start = max(start, stop - rise)
        env[tail_start:stop] *= ramp[:stop - tail_start][::-1]
    return env


def _chirp(f_start, f_stop, envelope=None):
    t = _time_axis()
    duration = N_COMPLEX / SAMPLE_RATE
    rate = (f_stop - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * rate * t ** 2)
    z = np.exp(1j * phase)
    return z if envelope is None else z * envelope


def _radar_altimeter(rng):
    """Continuous FMCW sweep, constant envelope."""
    span = rng.uniform(1.2e6, 2.2e6)
    center = rng.uniform(-0.6e6, 0.6e6)
    return _chirp(center - span / 2, center + span / 2)


def _satcom(rng):
    """Continuous QPSK with pulse shaping, near-constant envelope."""
    symbols = rng.integers(0, 4, size=16)
    constellation = np.exp(1j * (np.pi / 4 + symbols * np.pi / 2))
    waveform = np.repeat(constellation, N_COMPLEX // len(symbols))
    kernel = np.hanning(9)
    kernel /= kernel.sum()
    shaped = np.convolve(waveform, kernel, mode="same")
    carrier = np.exp(2j * np.pi * rng.uniform(-0.4e6, 0.4e6) * _time_axis())
    return shaped * carrier


def _short_range(rng):
    """ASK burst: amplitude keyed on and off around a narrow carrier."""
    symbols = rng.integers(0, 2, size=8).astype(float)
    symbols = 0.25 + 0.75 * symbols
    envelope = np.repeat(symbols, N_COMPLEX // len(symbols))
    kernel = np.hanning(7)
    kernel /= kernel.sum()
    envelope = np.convolve(envelope, kernel, mode="same")
    carrier = np.exp(2j * np.pi * rng.uniform(-0.3e6, 0.3e6) * _time_axis())
    return envelope * carrier


def _airborne_detection(rng):
    """Surveillance radar: narrowband pulse, moderate duty, single carrier."""
    width = int(rng.integers(18, 25))
    start = int(rng.integers(0, N_COMPLEX - width))
    envelope = _pulse_envelope(start, width)
    carrier = np.exp(2j * np.pi * rng.uniform(-1.5e6, 1.5e6) * _time_axis())
    return envelope * carrier


def _airborne_range(rng):
    """
    Range-finding radar: polyphase-coded pulse compression.

    A constant-modulus pseudo-random phase code spreads the pulse across the
    whole passband, which is what buys the range resolution. Spectrally it
    looks close to noise; the pulsed envelope is what separates it from a
    barrage jammer.
    """
    width = int(rng.integers(22, 31))
    start = int(rng.integers(0, N_COMPLEX - width))
    envelope = _pulse_envelope(start, width, rise=2)
    code = np.exp(2j * np.pi * rng.random(N_COMPLEX))
    return code * envelope


def _air_ground_mti(rng):
    """Moving target indicator: very short, very sharp pulse."""
    width = int(rng.integers(4, 8))
    start = int(rng.integers(0, N_COMPLEX - width))
    envelope = _pulse_envelope(start, width, rise=1)
    carrier = np.exp(2j * np.pi * rng.uniform(-2.0e6, 2.0e6) * _time_axis())
    return envelope * carrier


def _ew_jammer(rng):
    """Barrage jamming: band-limited complex noise filling the passband."""
    noise = rng.normal(size=N_COMPLEX) + 1j * rng.normal(size=N_COMPLEX)
    spectrum = np.fft.fft(noise)
    mask = np.ones(N_COMPLEX)
    notch = int(rng.integers(4, 10))
    mask[N_COMPLEX // 2 - notch:N_COMPLEX // 2 + notch] *= rng.uniform(0.3, 0.8)
    return np.fft.ifft(spectrum * mask)


def _am_radio(rng):
    """Commercial AM: narrow carrier with slow double-sideband modulation."""
    t = _time_axis()
    depth = rng.uniform(0.35, 0.55)
    tone = rng.uniform(6e4, 1.4e5)
    envelope = 1.0 + depth * np.cos(2 * np.pi * tone * t + rng.uniform(0, 2 * np.pi))
    carrier = np.exp(2j * np.pi * rng.uniform(-1.5e5, 1.5e5) * t)
    return envelope * carrier


_GENERATORS = {
    "Radar-Altimeter": _radar_altimeter,
    "Satcom": _satcom,
    "short-range": _short_range,
    "Airborne-detection": _airborne_detection,
    "Airborne-range": _airborne_range,
    "Air-Ground-MTI": _air_ground_mti,
    "EW-Jammer": _ew_jammer,
    "AM radio": _am_radio,
}


def build(signal_type, rng=None):
    """Build one clean complex waveform for the given emitter type."""
    if signal_type not in _GENERATORS:
        raise KeyError(f"unknown signal type: {signal_type}")
    if rng is None:
        rng = np.random.default_rng()
    return _GENERATORS[signal_type](rng)


def _channel(z, rng):
    """
    Per-receiver propagation effects: one delayed multipath echo and a small
    residual carrier offset. Each receiver sees its own channel, which is why
    this is applied here rather than when the waveform is built.
    """
    delay = int(rng.integers(1, 5))
    gain = rng.uniform(0.12, 0.42) * np.exp(2j * np.pi * rng.random())
    echo = np.zeros_like(z)
    echo[delay:] = z[:-delay]
    z = z + gain * echo

    offset = rng.normal(0.0, 1.2e4)
    return z * np.exp(2j * np.pi * offset * _time_axis())


def observe(z, snr_db=20.0, rng=None):
    """
    Push a clean waveform through a receiver's channel, add thermal noise, and
    pack it into a 256-element snapshot. One emission observed by several
    receivers shares the same z, so the snapshots stay correlated the way real
    co-observations are.
    """
    if rng is None:
        rng = np.random.default_rng()

    z = _channel(z, rng)
    signal_power = np.mean(np.abs(z) ** 2)
    if signal_power <= 0:
        signal_power = 1e-12

    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noisy = z + np.sqrt(noise_power / 2.0) * (
        rng.normal(size=N_COMPLEX) + 1j * rng.normal(size=N_COMPLEX)
    )

    scale = np.sqrt(np.mean(np.abs(noisy) ** 2))
    if scale > 0:
        noisy = noisy / scale

    return np.concatenate([noisy.real, noisy.imag]).astype(np.float32)


def generate(signal_type, snr_db=20.0, rng=None):
    """Build one 256-element IQ snapshot for the given emitter type."""
    if rng is None:
        rng = np.random.default_rng()
    return observe(build(signal_type, rng), snr_db, rng)
