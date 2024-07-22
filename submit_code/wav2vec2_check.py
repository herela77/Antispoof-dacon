import pandas as pd
import os

csv_file_path = './new_zero.csv'

audio_directory = './test_wav2vec2'

df = pd.read_csv(csv_file_path)

audio_files = os.listdir(audio_directory)

updated_df = df.copy()

for file_name in audio_files:
    if file_name.endswith('.ogg'):
        parts = file_name.split('_')
        if len(parts) > 2 and parts[2] == 'non':
            base_id = f"{parts[0]}_{parts[1]}"
            if base_id in updated_df['id'].values:
                updated_df.loc[updated_df['id'] == base_id, ['fake', 'real']] = 0

updated_df.to_csv('./wav2vec2.csv', index=False)

print("CSV 파일이 업데이트 되었습니다.")