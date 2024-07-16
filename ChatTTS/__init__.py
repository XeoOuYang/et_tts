import sys

from et_dirs import chattts_base
run_dir = chattts_base
if run_dir in sys.path:
    sys.path.remove(run_dir)
sys.path.insert(0, run_dir)
