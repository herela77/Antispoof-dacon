# Antispoof-dacon
---
### train
---
##### Supervised fine-tuning
python submit_doce/main.py --model ast --save_dir ast_bigbatch --batch_size 2048 --lr 0.001
##### Unlabeled fine-tuning
python baseline_code/main.py --model ast --save_dir ast_bigbatch_37epoch_unlabeld_thres0.9_withoutscheduler_afternew_sgd0.00001 --lr 0.00001 --weight_decay 0.001 --unlabel_ft --epochs 10 --batch_size 64
---
### inference
python submit_doce/test.py --sp submit.csv
