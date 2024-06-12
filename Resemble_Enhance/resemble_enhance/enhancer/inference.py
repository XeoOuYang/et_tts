import logging
from functools import cache
from pathlib import Path

import torch

from ..inference import inference
from .download import download
from .train import Enhancer, HParams

logger = logging.getLogger(__name__)
resemble_enhance = None

@cache
def load_enhancer(run_dir: str | Path | None, device):
    global resemble_enhance
    if resemble_enhance is None:
        run_dir = download(run_dir)
        hp = HParams.load(run_dir)
        resemble_enhance = Enhancer(hp)
        path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
        state_dict = torch.load(path, map_location=device)["module"]
        resemble_enhance.load_state_dict(state_dict)
        resemble_enhance.eval()
        resemble_enhance.to(device)
    # 返回结果
    return resemble_enhance


@torch.inference_mode()
def denoise(dwav, sr, device, run_dir=None):
    enhancer = load_enhancer(run_dir, device)
    return inference(model=enhancer.denoiser, dwav=dwav, sr=sr, device=device)


@torch.inference_mode()
def enhance(dwav, sr, device, nfe=32, solver="midpoint", lambd=0.5, tau=0.5, run_dir=None):
    assert 0 < nfe <= 128, f"nfe must be in (0, 128], got {nfe}"
    assert solver in ("midpoint", "rk4", "euler"), f"solver must be in ('midpoint', 'rk4', 'euler'), got {solver}"
    assert 0 <= lambd <= 1, f"lambd must be in [0, 1], got {lambd}"
    assert 0 <= tau <= 1, f"tau must be in [0, 1], got {tau}"
    enhancer = load_enhancer(run_dir, device)
    enhancer.configurate_(nfe=nfe, solver=solver, lambd=lambd, tau=tau)
    return inference(model=enhancer, dwav=dwav, sr=sr, device=device)
