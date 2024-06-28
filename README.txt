0、conda
conda create -n openvoice python=3.10
conda activate openvoice
conda install ffmpeg

1、install requirement.txt
(.venv)> pip install -r OpenVoice/requirement.txt
(.venv)> pip install -r MeloTTS/requirement.txt

2.1、install OpenVoiceV2
git clone git@github.com:myshell-ai/OpenVoice.git
cd OpenVoice
pip install -e .

2.2、install MeloTTS
git clone git@github.com:myshell-ai/MeloTTS.git
cd MeloTTS
pip install -e .
python -m unidic download

3、check cuda installation
import torch
print(torch.cuda.is_available())
if not available, please install it(ref:https://pytorch.org/get-started/locally/)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

4、FAQ
0)、install setup tool
pip install setuptools-rust
a)、ModuleNotFoundError: No module named 'whisper'
pip uninstall openai-whisper
pip install git+https://github.com/openai/whisper.git
b)、RuntimeError: Library cublas64_11.dll is not found or cannot be loaded
copy cublas64_12.dll and rename to cublas64_11.dll

5、install GPT-SoVits
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -e .

6、install rvc
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion
cd Retrieval-based-Voice-Conversion
pip install -e .

7、install so-vits-svc
git clone https://github.com/svc-develop-team/so-vits-svc
cd Retrieval-based-Voice-Conversion
pip install -e .

8、安装cuda
https://developer.nvidia.com/cuda-toolkit

9、安装conda
https://developer.nvidia.com/cuda-toolkit

10、版本纠正
pip install huggingface-hub==0.19.3
pip install transformers -U
pip install typing_extensions==4.9.0
conda install uvicorn fastapi
pip install librosa

11、flash-attn
pip install flash-attn
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers --index-url https://download.pytorch.org/whl/cu121 --upgrade

conda install -y -k cuda==12.1.1 -c nvidia/label/cuda-12.1.1
python -m pip install torch==2.1.0+cu121 torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu121

pip install https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl


12、数字转单词
pip install num2words

13、https://update.glados-config.com/clash/335545/e96a559/42774/glados.yaml

14、deepspeed
https://www.piwheels.org/project/deepspeed/

mkdir -p ~/.pip && echo -e "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" > ~/.pip/pip.conf
ffmpeg -i 88795527-1-208.mp3 -af silenceremove=start_periods=1:start_duration=1.0:start_threshold=-30dB:stop_periods=-1.0:stop_duration=1.0:stop_threshold=-30dB -y 88795527.mp3