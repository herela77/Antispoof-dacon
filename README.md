# Antispoof-dacon

---
# Project Directory Structure
```plaintext
main/
├── submit_code/
│   ├── main.py
│   ├── test.py
│   └── *.py
├── train/
│   ├── real.ogg
│   └── fake.ogg
├── test/
│   ├── Test_00001.ogg
│   ├── Test_00002.ogg
│   └── Test_00003.ogg
├── no_voice_overlay/
│   ├── combined_1.ogg
│   ├── combined_3.ogg
│   └── combined_27600.ogg
└── novoice/
    ├── ABJGMLHQ_accompaniment.ogg
    ├── ABKEEJML_accompaniment.ogg
    └── ESNQMTYC_accompaniment.ogg
```
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
## Inference
```bash
python submit_doce/test.py --sp submit.csv
```
