import sys

from et_dirs import melo_tts_base

run_dir = melo_tts_base
if run_dir in sys.path:
    sys.path.remove(run_dir)
sys.path.insert(0, run_dir)
