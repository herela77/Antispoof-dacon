import os
import random
from pydub import AudioSegment
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 랜덤 시드 고정
random.seed(42)

audio_folder = "./novoice"
output_folder = "./no_voice_overlay"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.ogg')]

# 생성할 오디오 파일 수
num_generated_files = 27600
# 오버레이에 사용할 파일 수
num_files_to_overlay = 4  # 원하는 개수로 조정

def overlay_random_audios(index):
    selected_files = random.sample(audio_files, num_files_to_overlay)
    combined = AudioSegment.from_file(os.path.join(audio_folder, selected_files[0]))
    
    for file in selected_files[1:]:
        next_audio = AudioSegment.from_file(os.path.join(audio_folder, file))
        combined = combined.overlay(next_audio)
        
    output_path = os.path.join(output_folder, f"combined_{index+1}.ogg")
    combined.export(output_path, format="ogg")

if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(overlay_random_audios, range(num_generated_files)), total=num_generated_files))
