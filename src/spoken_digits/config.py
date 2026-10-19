from pathlib import Path
import os


# Dataset
PROJECT_ROOT = Path(
    os.environ.get("SPOKEN_DIGITS_ROOT", Path.cwd())
).resolve()

RECORDING_DIR = PROJECT_ROOT / "data" / "free-spoken-digit-dataset" / "recordings"

# Sampling frequency
FS = 8000   # all FSDD recordings are sampled at 8 kHz

# MFCC
WINDOW_LENGTH_SECONDS = 0.025
HOP_LENGTH_SECONDS = 0.015
WINDOW_LENGTH = round(WINDOW_LENGTH_SECONDS * FS)
HOP_LENGTH = round(HOP_LENGTH_SECONDS * FS)

N_FFT = 1024

USE_VAD = True
USE_PREEMPHASIS = True

# GMM
N_COMPONENTS = 5
SEED = 42