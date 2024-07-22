import argparse
import torch
import pandas as pd
from dataloader import AudioDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from ast_model import ASTModel
import os
from util import Test
warnings.simplefilter(action='ignore', category=FutureWarning)



parser = argparse.ArgumentParser()
parser.add_argument('--sp', default='', type=str, help='save dir')
parser.add_argument('--pth', default='', type=str, help='save dir')
args = parser.parse_args()


# 데이터 로드 및 전처리
test = pd.read_csv('./test.csv')
test_dataset = AudioDataset(test, train_mode=False)
test_loader = DataLoader(
    test_dataset,
    batch_size=5000,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)


# 모델 정의 및 로드

input_fdim = 84
input_tdim = 100
label_dim = 2
model = ASTModel(input_fdim=input_fdim, input_tdim=input_tdim, label_dim=2, audioset_pretrain=True)

model = torch.nn.DataParallel(model).cuda()
state_dict = torch.load("ast_bigbatch_37epoch_unlabeld_thres0.9_withoutscheduler_afternew_sgd0.00001/1_0.12760130451449958test_auc.pth")

model.load_state_dict(state_dict)


# 예측 수행
os.makedirs("test_result", exist_ok=True)
tester = Test(
    model=model,
    test_loader=test_loader,
    submission_path='./sample_submission.csv',
    zero_csv1_path='bert_zero_5921.csv',
    zero_csv2_path='zero_6653.csv',
    save_path=args.sp
)
predictions = tester.inference()
tester.result_zero_cover(predictions)
