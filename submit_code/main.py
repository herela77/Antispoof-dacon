import argparse
from dataloader import AudioDataset
from dataloader_unlabeled import AudioDataset_unlabeled 
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import torch
import os
import train
import train_unlabeled 
import val
import glob
from lcnn import LightCNN
import warnings
from ast_model import ASTModel


warnings.filterwarnings('ignore')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--SR', type=int, default=32000)
    parser.add_argument('--model', default='lcnn', type=str, help='save dir')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--N_CLASSES', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--save_dir', default='lcnn', type=str, help='save dir')
    parser.add_argument('--no_voice_dir', default='no_voice_overlay/', type=str, help='no_voice_dir')
    parser.add_argument('--unlabel_ft', action="store_true")

    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    seed_everything(args.seed) # Seed 고정
    df = pd.read_csv('./train.csv')
    df2 = pd.read_csv('./combined_audio_metadata.csv')

    test_df = pd.read_csv('./SOTA.csv')

    df2['label'] = list(zip(df2['fake'], df2['real']))
    df2 = df2[['path', 'label']]
    df['label'] = df['label'].apply(lambda x: [1, 0] if x == 'fake' else [0, 1])

    no_voice_paths = glob.glob(os.path.join(args.no_voice_dir, '*.ogg'))
    no_voice_df = pd.DataFrame({
        'path': no_voice_paths,
        'label': [[0, 0]] * len(no_voice_paths)
    })

    all_data = pd.concat([df, df2, no_voice_df], ignore_index=True)
    all_data['label'] = all_data['label'].apply(lambda x: tuple(x))
    train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=args.seed, stratify=all_data['label'])

    
    if args.unlabel_ft:
        train_loader = DataLoader(
            AudioDataset_unlabeled(),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers = 24
        )
    else:
        train_loader = DataLoader(
            AudioDataset(train_data),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers = 24
        )
    val_loader = DataLoader(
        AudioDataset(val_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers = 24
    )
    
    if args.model =='lcnn':
        model = LightCNN()
    elif args.model =='ast':
        input_fdim = 84
        input_tdim = 100
        label_dim = 2
        model = ASTModel(input_fdim=input_fdim, input_tdim=input_tdim, label_dim=2, audioset_pretrain=True)
    elif args.model =='':
        pass


    #### distributed 
    n_gpus = torch.cuda.device_count()

    print('Number of GPUs:', torch.cuda.device_count())

    model = torch.nn.parallel.DataParallel(model).cuda()
    if args.unlabel_ft:
        state_dict = torch.load("ast_bigbatch/ast_bigbatch37_0.12741267722824998test_auc.pth")
        model.load_state_dict(state_dict)


    criterion = torch.nn.MultiLabelSoftMarginLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.unlabel_ft:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)

    best_score = 100.0
    best_test_score = 100.0
    for epoch in range(1, args.epochs+1):
        if args.unlabel_ft:
            train_unlabeled.train_unlabel(model, optimizer, criterion, train_loader, epoch)
        else:
            train.train(model, optimizer, criterion, train_loader, scheduler, epoch)
        best_model, best_test_score = val.validation(model, criterion, val_loader)

        if best_model is not None :
            epoch_save_name = os.path.join(args.save_dir, f'{epoch}_{best_test_score}test_auc.pth')
            torch.save(model.state_dict(), epoch_save_name)
