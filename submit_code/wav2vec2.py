from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch
import glob
import os
import shutil

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

input_audio_path_pattern = './mini_test/*.ogg'
output_directory = './test_wav2vec2'

os.makedirs(output_directory, exist_ok=True)

audio_files = glob.glob(input_audio_path_pattern)

def process_audio_file(audio_file):
    base_name = os.path.basename(audio_file)
    name, ext = os.path.splitext(base_name)
    
    audio_input, sample_rate = torchaudio.load(audio_file)

    if sample_rate != 16000:
       resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
       audio_input = resampler(audio_input)
       sample_rate = 16000

    audio_input = audio_input.mean(dim=0)

    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
    transcription = processor.decode(predicted_ids[0])
    
    if len(transcription.strip()) == 0:
        new_file_name = f"{name}_non_speech_{len(transcription.strip())}{ext}"
    else:
        new_file_name = f"{name}_speech{len(transcription.strip())}{ext}"
    
    new_file_path = os.path.join(output_directory, new_file_name)
    shutil.copy(audio_file, new_file_path)
    print(f"Processed {audio_file}: {'speech' if len(transcription.strip()) > 0 else 'non_speech'} -> {new_file_path}")

for audio_file in audio_files:
    process_audio_file(audio_file)
