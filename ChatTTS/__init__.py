import sys

from et_dirs import gpt_sovits_base
run_dir = gpt_sovits_base
if run_dir in sys.path:
    sys.path.remove(run_dir)
sys.path.insert(0, run_dir)
