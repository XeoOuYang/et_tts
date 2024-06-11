import glob
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import librosa
import numpy as np
import soundfile
import torch

from inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

model = None
spk = None
debug = False

from SoVITS_SVC.sovits_svc_tts import base_dir
local_model_root = './trained'

cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"


def modelAnalysis(model_path, config_path, cluster_model_path, device, enhance, diff_model_path, diff_config_path, only_diffusion, use_spk_mix,
                  local_model_enabled, local_model_selection):
    global model
    try:
        device = cuda[device] if "CUDA" in device else device
        cluster_filepath = os.path.split(cluster_model_path.name) if cluster_model_path is not None else "no_cluster"
        # get model and config path
        if (local_model_enabled):
            # local path
            model_path = glob.glob(os.path.join(local_model_selection, '*.pth'))[0]
            config_path = glob.glob(os.path.join(local_model_selection, '*.json'))[0]
        else:
            # upload from webpage
            model_path = model_path
            config_path = config_path
        fr = ".pkl" in cluster_filepath[1]
        model = Svc(model_path,
                    config_path,
                    device=device if device != "Auto" else None,
                    cluster_model_path=cluster_model_path.name if cluster_model_path is not None else "",
                    nsf_hifigan_enhance=enhance,
                    diffusion_model_path=diff_model_path.name if diff_model_path is not None else "",
                    diffusion_config_path=diff_config_path.name if diff_config_path is not None else "",
                    shallow_diffusion=True if diff_model_path is not None else False,
                    only_diffusion=only_diffusion,
                    spk_mix_enable=use_spk_mix,
                    feature_retrieval=fr
                    )
        spks = list(model.spk2id.keys())
        device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
        msg = f"成功加载模型到设备{device_name}上\n"
        if cluster_model_path is None:
            msg += "未加载聚类模型或特征检索模型\n"
        elif fr:
            msg += f"特征检索模型{cluster_filepath[1]}加载成功\n"
        else:
            msg += f"聚类模型{cluster_filepath[1]}加载成功\n"
        if diff_model_path is None:
            msg += "未加载扩散模型\n"
        else:
            msg += f"扩散模型{diff_model_path.name}加载成功\n"
        msg += "当前模型的可用音色：\n"
        for i in spks:
            msg += i + " "
        return spks[0], msg
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise e


def modelUnload():
    global model
    if model is None:
        return "没有模型需要卸载!"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return "模型卸载完毕!"


def vc_infer(output_format, sid, audio_path, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds,
             cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding,
             loudness_envelope_adjustment):
    global model
    _audio = model.slice_inference(
        audio_path,
        sid,
        vc_transform,
        slice_db,
        cluster_ratio,
        auto_f0,
        noise_scale,
        pad_seconds,
        cl_num,
        lg_num,
        lgr_num,
        f0_predictor,
        enhancer_adaptive_key,
        cr_threshold,
        k_step,
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment
    )
    model.clear_empty()
    # 构建保存文件的路径，并保存到results文件夹内
    str(int(time.time()))
    result_dir = os.path.join(base_dir, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    key = "auto" if auto_f0 else f"{int(vc_transform)}key"
    cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
    isdiffusion = "sovits"
    if model.shallow_diffusion:
        isdiffusion = "sovdiff"

    if model.only_diffusion:
        isdiffusion = "diff"

    output_file_name = truncated_basename + f'_{sid}_{key}{cluster}{isdiffusion}.{output_format}'
    output_file = os.path.join(result_dir, output_file_name)
    soundfile.write(output_file, _audio, model.target_sample, format=output_format)
    return output_file


def vc_fn(input_audio, sid, output_format, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num,
          f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    global model
    try:
        if input_audio is None:
            return "You need to upload an audio", None
        if model is None:
            return "You need to upload an model", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False and cluster_ratio != 0:
            return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        # print(input_audio)
        audio, sampling_rate = soundfile.read(input_audio)
        # print(audio.shape,sampling_rate)
        if np.issubdtype(audio.dtype, np.integer):
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        # print(audio.dtype)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        # 未知原因Gradio上传的filepath会有一个奇怪的固定后缀，这里去掉
        truncated_basename = Path(input_audio).stem[:-6]
        processed_audio = os.path.join("raw", f"{truncated_basename}.wav")
        soundfile.write(processed_audio, audio, sampling_rate, format="wav")
        output_file = vc_infer(output_format, sid, processed_audio, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db,
                               noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step,
                               use_spk_mix, second_encoding, loudness_envelope_adjustment)

        return "Success", output_file
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise e


def vc_fn2(_text, _lang, _gender, _rate, _volume, sid, output_format, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds,
           cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    global model
    try:
        if model is None:
            return "You need to upload an model", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False and cluster_ratio != 0:
            return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        _rate = f"+{int(_rate * 100)}%" if _rate >= 0 else f"{int(_rate * 100)}%"
        _volume = f"+{int(_volume * 100)}%" if _volume >= 0 else f"{int(_volume * 100)}%"
        from et_dirs import sovits_svc_base
        py_cmd = os.path.join(sovits_svc_base, "edgetts/tts.py")
        if _lang == "Auto":
            _gender = "Male" if _gender == "男" else "Female"
            subprocess.run([sys.executable, py_cmd, _text, _lang, _rate, _volume, _gender])
        else:
            subprocess.run([sys.executable, py_cmd, _text, _lang, _rate, _volume])
        target_sr = 44100
        tts_result = os.path.join(sovits_svc_base, f'results{os.path.sep}tts.wav')
        y, sr = librosa.load(tts_result)
        resampled_y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        soundfile.write(tts_result, resampled_y, target_sr, subtype="PCM_16")
        input_audio = tts_result
        # audio, _ = soundfile.read(input_audio)
        output_file_path = vc_infer(output_format, sid, input_audio, "tts", vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale,
                                    pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step,
                                    use_spk_mix, second_encoding, loudness_envelope_adjustment)
        os.remove(tts_result)
        return "Success", output_file_path
    except Exception as e:
        if debug: traceback.print_exc()  # noqa: E701
        raise e


def vc_fn3(input_audio, sid, output_format, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds,
           cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    global model
    try:
        if model is None:
            return "You need to upload an model", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False and cluster_ratio != 0:
            return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        target_sr = 44100
        y, sr = librosa.load(input_audio)
        resampled_y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        soundfile.write(input_audio, resampled_y, target_sr, subtype="PCM_16")
        input_audio = input_audio
        # audio, _ = soundfile.read(input_audio)
        output_file_path = vc_infer(output_format, sid, input_audio, "tts", vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale,
                                    pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step,
                                    use_spk_mix, second_encoding, loudness_envelope_adjustment)
        # os.remove(input_audio)
        return "Success", output_file_path
    except Exception as e:
        if debug: traceback.print_exc()  # noqa: E701
        raise e
