import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import glob
import os
import shutil
from multiprocessing import Pool, cpu_count

vad_model = load_silero_vad()
sr = 16000

input_audio_path_pattern = './test/*.ogg'
output_directory = './test_silero'

os.makedirs(output_directory, exist_ok=True)

audio_files = glob.glob(input_audio_path_pattern)

def process_audio_file(audio_file):
    base_name = os.path.basename(audio_file)
    
    wav = read_audio(audio_file, sampling_rate=sr)
    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=sr)
    name, ext = os.path.splitext(base_name)
    if len(speech_timestamps) == 0:
        new_file_name = f"{name}non_speech{ext}"
    else:
        new_file_name = f"{name}speech{ext}"
    new_file_path = os.path.join(output_directory, new_file_name)
    shutil.copy(audio_file, new_file_path)
    print(f"Copied and renamed file to {new_file_path}")


if __name__ == "__main__":
    pool = Pool(cpu_count())
    pool.map(process_audio_file, audio_files)
    pool.close()
    pool.join()
    
    
