import sys

from et_dirs import prj_dir, llm_ai_base, melo_tts_base, ov_v2_base
from et_dirs import gpt_sovits_base,  rvc_cvt_base, sovits_svc_base, suno_bark_base, chattts_base

if prj_dir in sys.path:
    sys.path.remove(prj_dir)
sys.path.insert(0, prj_dir)
sys.path.append(gpt_sovits_base)
sys.path.append(llm_ai_base)
sys.path.append(melo_tts_base)
sys.path.append(ov_v2_base)
sys.path.append(rvc_cvt_base)
sys.path.append(sovits_svc_base)
sys.path.append(suno_bark_base)
sys.path.append(chattts_base)
