import sys

from et_dirs import ov_v2_base
run_dir = ov_v2_base
if run_dir in sys.path:
    sys.path.remove(run_dir)
sys.path.insert(0, run_dir)
