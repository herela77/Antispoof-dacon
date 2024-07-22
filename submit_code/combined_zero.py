import pandas as pd

file_1 = pd.read_csv('./wav2vec2.csv')
file_2 = pd.read_csv('./silero.csv')

if len(file_1) != len(file_2):
    raise ValueError("The two files must have the same number of rows")

def remove_second_underscore(id_value):
    parts = id_value.split('_')
    if len(parts) > 2:
        return '_'.join(parts[:2]) + ''.join(parts[2:])
    return id_value

combined_list = []

for idx in range(len(file_1)):
    row_id = remove_second_underscore(file_1.loc[idx, 'id'])
    if (file_1.loc[idx, 'fake'] == 0 and file_1.loc[idx, 'real'] == 0) or \
       (file_2.loc[idx, 'fake'] == 0 and file_2.loc[idx, 'real'] == 0):
        combined_list.append({'id': row_id, 'fake': 0, 'real': 0})
    else:
        combined_list.append({'id': row_id, 'fake': 1, 'real': 1})

combined_df = pd.DataFrame(combined_list)

output_file_combined = './zero_6653.csv'
combined_df.to_csv(output_file_combined, index=False)

print(f"Combined file saved to {output_file_combined}")
