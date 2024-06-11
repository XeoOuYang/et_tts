import os
import sys

from et_dirs import rvc_cvt_base
run_dir = rvc_cvt_base
if run_dir in sys.path:
    sys.path.remove(run_dir)
sys.path.insert(0, run_dir)
