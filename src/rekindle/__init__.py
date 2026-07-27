"""Rekindle: time-aware two-stage product discovery."""

import os

# On Apple Silicon, importing LightGBM's OpenMP runtime before PyTorch can crash the process
# during Torch CPU operations. The neural path uses MPS, and the local ranker is single-worker,
# so a process-wide one-thread OpenMP cap is the stable, intentional configuration.
os.environ["OMP_NUM_THREADS"] = "1"

__version__ = "0.1.0"
