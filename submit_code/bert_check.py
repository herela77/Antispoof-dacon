import pandas as pd
import os

# CSV 파일 경로
csv_file_path = './new_zero.csv'

# 오디오 파일들이 있는 디렉토리 경로
audio_directory = './test_bert'

# CSV 파일 읽기
df = pd.read_csv(csv_file_path)

# 오디오 파일 목록 가져오기
audio_files = os.listdir(audio_directory)

# 변경된 값을 저장할 새로운 DataFrame 생성
updated_df = df.copy()

# 오디오 파일 처리
for file_name in audio_files:
    if file_name.endswith('.ogg'):
        parts = file_name.split('_')
        if len(parts) > 2 and parts[2] == 'non':
            base_id = f"{parts[0]}_{parts[1]}"
            if base_id in updated_df['id'].values:
                updated_df.loc[updated_df['id'] == base_id, ['fake', 'real']] = 0
        elif int(parts[2].split('h')[1].split('.')[0]) <= 3:
            base_id = f"{parts[0]}_{parts[1]}"
            if base_id in updated_df['id'].values:
                updated_df.loc[updated_df['id'] == base_id, ['fake', 'real']] = 0

# CSV 파일 저장
updated_df.to_csv('./bert_zero_5921.csv', index=False)

print("CSV 파일이 업데이트 되었습니다.")