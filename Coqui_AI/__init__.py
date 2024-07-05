import sys

from et_dirs import coqui_ai_base
run_dir = coqui_ai_base
if run_dir in sys.path:
    sys.path.remove(run_dir)
sys.path.insert(0, run_dir)
