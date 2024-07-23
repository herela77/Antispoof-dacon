import pandas as pd
from pydub import AudioSegment
import os
import random
from tqdm import tqdm

os.makedirs("sample_audio", exist_ok=True)

data = pd.read_csv('./train.csv')
df = pd.DataFrame(data)

def overlay_and_save(audio_path1, audio_path2, output_path):
    audio1 = AudioSegment.from_file(audio_path1)
    audio2 = AudioSegment.from_file(audio_path2)
    combined = audio1.overlay(audio2)
    combined.export(output_path, format="mp3")

count = 27600

real_paths = df[df['label'] == 'real']['path'].tolist()
fake_paths = df[df['label'] == 'fake']['path'].tolist()

combinations = [
    ('real', 'real', real_paths, real_paths, [0, 1]),
    ('fake', 'real', fake_paths, real_paths, [1, 1]),
    ('fake', 'fake', fake_paths, fake_paths, [1, 0])
]

results = []

for label1, label2, paths1, paths2, label_encoding in combinations:
    for i in tqdm(range(count)):
        index1 = random.randint(0, len(paths1) - 1)
        index2 = random.randint(0, len(paths2) - 1)
        audio_path1 = paths1[index1]
        audio_path2 = paths2[index2]
        output_filename = f"{label1}_{label2}_{i}.ogg"
        output_path = os.path.join("sample_audio", output_filename)
        overlay_and_save(audio_path1, audio_path2, output_path)
        output_filename = f"./sample_audio/{label1}_{label2}_{i}.ogg"
        results.append([output_filename] + label_encoding)

results_df = pd.DataFrame(results, columns=["path", "fake", "real"])

results_df.to_csv("./combined_audio_metadata.csv", index=False)
print("CSV file with metadata has been created.")
