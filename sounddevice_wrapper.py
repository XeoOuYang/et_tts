import os.path

import sounddevice_v2 as sd
import soundfile as sf

import subprocedure


def list_mme():
    mme_item = [item for item in sd.query_hostapis() if item['name'] == 'MME']
    assert len(mme_item) > 0
    mme_item = mme_item[0]
    if mme_item:
        device_list = mme_item['devices']
        device_list = [sd.query_devices(device=device) for device in device_list]
        device_list = [item for item in device_list if item['max_output_channels'] > 0]
        device_list = [item for item in device_list if 'Microsoft' not in item['name']
                       and 'NVIDIA' not in item['name'] and 'Digital' not in item['name']]
        for device in device_list:
            beep_bu(device['index'], device['name'])


def name_to_index(name_list, silent=False):
    mme_item = [item for item in sd.query_hostapis() if item['name'] == 'MME']
    assert len(mme_item) > 0
    mme_item = mme_item[0]
    if mme_item:
        device_list = mme_item['devices']
        device_list = [sd.query_devices(device=device) for device in device_list]
        device_list = [item for item in device_list if item['max_output_channels'] > 0]
        device_list = [item for item in device_list if 'Microsoft' not in item['name']
                       and 'NVIDIA' not in item['name'] and 'Digital' not in item['name']]
        device_list = {item['name']: item['index'] for item in device_list if item['name'] in name_list}
        index_list = [device_list[k] if k in device_list else -1 for k in name_list]
        assert len(name_list) == len(index_list)
        if not silent:
            for idx, name in zip(index_list, name_list):
                beep_bu(idx, name)
        return index_list
    else:
        if not silent:
            for name in name_list:
                beep_bu(name, name)
        return name_list


def beep_bu(device, name):
    import numpy as np
    samplerate = 44100
    duration = 3
    arr = np.arange(samplerate * duration)
    arr = np.sin(2 * np.pi * 440 / samplerate * arr)
    from et_base import timer
    print('=='*10, f'{device}-{name}', '=='*10)
    with timer(f'{device}-{name}'):
        sd.play(arr, samplerate, device=device)
        sd.wait()


SOUND_DEVICE_NAME = ['扬声器 (High Definition Audio Devi', 'Headphones (High Definition Aud']
SOUND_DEVICE_INDEX = name_to_index(SOUND_DEVICE_NAME, True)
SOUND_DEVICE = {name: index for name, index in zip(SOUND_DEVICE_NAME, SOUND_DEVICE_INDEX)}


def play_wav_on_device(wav, device):
    # 查询设备idx
    if isinstance(device, str) and device in SOUND_DEVICE:
        device = SOUND_DEVICE[device]
    assert sd.query_devices(device=device, kind='output')
    # 设备存在
    if os.path.exists(wav):
        data, fs = sf.read(wav, dtype='float32')
        sd.play(data, fs, device=device)
        sd.wait()


def play_audio_async(wav, device):
    """
    # wav：音频地址
    # device：指定播放设备，参考sounddevice_wrapper.list_mme函数获取输出设备
    """
    procedure = subprocedure.SubProcedure(target=play_wav_on_device, kwargs={"wav": wav, "device": device})
    procedure.start()
    return procedure


def _play_audio_batch_(audio_list, device_list):
    """
    # audio_list：音频地址列表
    # device_list：播放设备列表，与音频地址列表一一对应
    """
    # 以下采用playsound播放声音,需要补充encoder编码信息
    # import playsound
    # try:
    #     playsound.playsound(audio_path)
    # except playsound.PlaysoundException:
    #     from et_base import fix_mci
    #     audio_path = fix_mci(audio_path)
    #     playsound.playsound(audio_path)
    # 以下采用sounddevice播放声音
    thread_list = []
    for wav, device in zip(audio_list, device_list):
        thread_list.append(subprocedure.SubProcedure(target=play_wav_on_device, kwargs={"wav": wav, "device": device}))
    # 启动线程
    for thread in thread_list:
        thread.start()
    # 等执行完成
    for thread in thread_list:
        thread.join()


if __name__ == '__main__':
    # audio_file = os.path.abspath(f'resources/man_role0_ref.wav')
    # data, fs = sf.read(audio_file, dtype='float32')
    # # 扬声器 (High Definition Audio Devi
    # # Headphones (High Definition Aud
    # sd.play(data, fs, device='扬声器 (High Definition Audio Devi')
    # sd.wait()
    list_mme()
