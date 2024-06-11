import sys

from et_dirs import suno_bark_base
run_dir = suno_bark_base
if run_dir in sys.path:
    sys.path.remove(run_dir)
sys.path.insert(0, run_dir)
