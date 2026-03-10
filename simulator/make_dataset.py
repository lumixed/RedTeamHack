"""
Writes the labelled training set the classifier trains on.

Keys follow the layout load_training_data() parses: the literal repr of a
(modulation, label, snr_db, chunk) tuple, with the label in position 1.
Only friendly emitters appear here - hostile and civilian types are meant to
reach the one-class detector as genuine out-of-distribution signals.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from .signals import FRIENDLY_TYPES, MODULATION_OF, generate

SNR_STEPS = list(range(6, 30, 3))
DEFAULT_PER_BUCKET = 160
OUT_PATH = Path(__file__).parent.parent / "data" / "training.h5"


def build(out_path=OUT_PATH, per_bucket=DEFAULT_PER_BUCKET, seed=42):
    rng = np.random.default_rng(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with h5py.File(out_path, "w") as f:
        for label in FRIENDLY_TYPES:
            modulation = MODULATION_OF[label]
            for snr in SNR_STEPS:
                block = np.stack([
                    generate(label, snr_db=snr + rng.uniform(-1.0, 1.0), rng=rng)
                    for _ in range(per_bucket)
                ])
                f.create_dataset(str((modulation, label, snr, 0)), data=block,
                                 compression="gzip", compression_opts=4)
                total += len(block)

    return out_path, total


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--per-bucket", type=int, default=DEFAULT_PER_BUCKET)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path, total = build(args.out, args.per_bucket, args.seed)
    size_mb = path.stat().st_size / 1e6
    print(f"Wrote {total} samples across {len(FRIENDLY_TYPES)} classes "
          f"and {len(SNR_STEPS)} SNR steps")
    print(f"  {path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
