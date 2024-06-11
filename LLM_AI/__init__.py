import sys

from et_dirs import llm_ai_base

run_dir = llm_ai_base
if run_dir in sys.path:
    sys.path.remove(run_dir)
sys.path.insert(0, run_dir)
