import sys

from et_dirs import sovits_svc_base

run_dir = sovits_svc_base
if run_dir in sys.path:
    sys.path.remove(run_dir)
sys.path.insert(0, run_dir)
