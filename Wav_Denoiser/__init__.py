import os
import sys

from et_dirs import wav_denoise_base
run_dir = wav_denoise_base
if run_dir in sys.path:
    sys.path.remove(run_dir)
sys.path.insert(0, run_dir)
