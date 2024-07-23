# Antispoof-dacon

---
## Project Directory Structure
```plaintext
main/
├── ast_bigbatch/
│   └── ast_bigbatch37_0.12741267722824998test_auc.pth
│
├── ast_bigbatch_37epoch_unlabeld_thres0.9_withoutscheduler_afternew_sgd0.00001/
│   └── 1_0.12760130451449958test_auc.pth
│
├── submit_code/
│   ├── main.py
│   ├── test.py
│   └── *.py
│
├── train/
│   ├── real.ogg
│   └── fake.ogg
│
├── sample_audio/
│   ├── real_real.ogg
│   ├── fake_fake.ogg
│   └── fake_real.ogg
│
├── test/
│   ├── Test_00001.ogg
│   ├── Test_00002.ogg
│   └── Test_00003.ogg
│
├── no_voice_overlay/
│   ├── combined_1.ogg
│   ├── combined_3.ogg
│   └── combined_27600.ogg
│
└── novoice/
│   ├── ABJGMLHQ_accompaniment.ogg
│   ├── ABKEEJML_accompaniment.ogg
│   └── ESNQMTYC_accompaniment.ogg
│
├── test_silero/
│
├── test_wav2vec2/
│
└── test_bert/
```
---
## development environment
#### Ubuntu 20.04.5 LTS
#### 8 A6000 GPUs
---
Sources of the models:

- Noise extractor source: https://github.com/deezer/spleeter/blob/master/spleeter/separator.py
- Silero source: https://github.com/snakers4/silero-vad
- Wav2Vec2 source: https://huggingface.co/facebook/wav2vec2-base-960h
- Wav2Vec2-BERT source: https://huggingface.co/tbkazakova/wav2vec-bert-2.0-even-pakendorf
- AST source: https://github.com/YuanGongND/ast

---
## To install the required packages, please use the following command with the provided `requirements.txt` file:
`pip install -r requirements.txt`
---

## Training Dataset Preparation

### To prepare the training dataset, follow these steps:

- **novoice**: Execute the script `no_voice_maker.py`
- **no_voice_overlay**: Execute the script `novoice_overlay.py`
-  **sample_audio**: Execute the script `make_sample_audio.py`

### You can also download the dataset and model weights from the following link: [Dataset and model Weight](https://drive.google.com/drive/folders/12Cmq278Q6p9a35BQ_TKXhV0cHouJCpF7?usp=drive_link)
---
## Training

### Supervised Fine-tuning

```bash
python submit_doce/main.py --model ast --save_dir ast_bigbatch --batch_size 2048 --lr 0.001
```
### Unlabeled Fine-tuning
```bash
python baseline_code/main.py --model ast --save_dir ast_bigbatch_37epoch_unlabeld_thres0.9_withoutscheduler_afternew_sgd0.00001 --lr 0.00001 --weight_decay 0.001 --unlabel_ft --epochs 10 --batch_size 64
```
---
## Directory Preparation

Before running the following scripts, make sure to create the specified directories and set the `input_audio_path_pattern` to the directory containing the test data.

### 1. Silero

- **Create Directory**: `test_silero`
- **Set `input_audio_path_pattern`**: Path to the test data

### 2. Wav2Vec2

- **Create Directory**: `test_wav2vec2`
- **Set `input_audio_path_pattern`**: Path to the test data

### 3. Wav2Vec2 BERT

- **Create Directory**: `test_bert`
- **Set `input_audio_path_pattern`**: Path to the test data

## Script Execution Order

Follow this sequence to execute the scripts properly:

1. **Run Silero Check**: `silero_check.py`
2. **Run Wav2Vec2 Check**: `wav2vec2_check.py`
3. **Run BERT Check**: `bert_check.py`
4. **Run Combined Zero**: `combined_zero.py`

Ensure each script is executed in the specified order to maintain consistency and correctness of the results.
---
## Inference
```bash
python submit_doce/test.py --sp submit.csv
```
