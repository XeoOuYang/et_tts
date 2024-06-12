import json
import os.path

import torch


class Spec_Voice:
    def __init__(self):
        self.speakers = {}

    def create_from_tensor(self, _id, tensor, gender="", describe=""):
        speaker = {"gender": gender, "describe": describe, "emb": None, }
        if isinstance(tensor, torch.Tensor):
            speaker['emb'] = tensor
        if isinstance(tensor, list):
            speaker['emb'] = torch.tensor(tensor)
        # 缓存起来
        self.speakers[_id] = speaker

    def get_by(self, seed):
        _id_ = f'{seed}'
        if _id_ in self.speakers:
            return self.speakers[_id_]['emb']
        else:
            return None


spec_voice = Spec_Voice()
from et_dirs import chattts_base
json_path = os.path.join(chattts_base, 'spec_voices', "voice_240605.json")
data = json.load(open(json_path, "r", encoding='utf8', errors='ignore'))

for _id, spk in data.items():
    spec_voice.create_from_tensor(
        _id=_id,
        tensor=spk["tensor"],
        gender=spk["gender"],
        describe=spk["describe"],
    )