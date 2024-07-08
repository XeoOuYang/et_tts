def clone_voice(ref_speaker):
    from bark.generation import load_codec_model
    from encodec.utils import convert_audio
    import torchaudio
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_codec_model(use_gpu=True)

    from hubert.hubert_manager import HuBERTManager
    hubert_manager = HuBERTManager()
    hubert_manager.make_sure_hubert_installed()
    hubert_manager.make_sure_tokenizer_installed()

    from hubert.pre_kmeans_hubert import CustomHubert
    from hubert.customtokenizer import CustomTokenizer

    # Load the HuBERT model
    hubert_model = CustomHubert(checkpoint_path='data/models/hubert/hubert.pt').to(device)
    # Load the CustomTokenizer model
    tokenizer = CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer.pth').to(device)  # Automatically uses the right layers

    # Load and pre-process the audio waveform
    audio_filepath = ref_speaker  # the audio you want to clone (under 13 seconds)
    wav, sr = torchaudio.load(audio_filepath)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.to(device)

    semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
    semantic_tokens = tokenizer.get_token(semantic_vectors)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav.unsqueeze(0))
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

    # move codes to cpu
    codes = codes.cpu().numpy()
    # move semantic tokens to cpu
    semantic_tokens = semantic_tokens.cpu().numpy()

    voice_name = 'man_0_es'  # whatever you want the name of the voice to be
    output_path = 'bark/assets/prompts/' + voice_name + '.npz'
    import numpy as np
    np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)

    from bark import SAMPLE_RATE, generate_audio
    from scipy.io import wavfile
    text_prompt = "Hello, my name is Serpy. And, uh — and I like pizza. [laughs]"
    audio_array = generate_audio(text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7)
    from et_dirs import outputs_v2
    import os.path
    save_path = f'{outputs_v2}{os.path.sep}output.wav'
    wavfile.write(save_path, SAMPLE_RATE, audio_array)