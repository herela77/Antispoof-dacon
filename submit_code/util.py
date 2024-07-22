import numpy as np
import torch
import pandas as pd

def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (targets, shuffled_targets, lam)
    
    return data, targets

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    
    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = (targets, shuffled_targets, lam)
    
    return data, targets

class Test:
    def __init__(self, model, test_loader, submission_path, zero_csv1_path, zero_csv2_path, save_path):
        self.model = model
        self.test_loader = test_loader
        self.submission_path = submission_path
        self.zero_csv1_path = zero_csv1_path
        self.zero_csv2_path = zero_csv2_path
        self.save_path = save_path

    def inference(self):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for features in self.test_loader:
                features = features.float().cuda()
                logits = self.model(features)
                probs = torch.sigmoid(logits)
                probs[:, 0] = probs[:, 0] + 0.033975
                probs[:, 1] = probs[:, 1] - 0.17343
                probs = torch.clamp(probs, 0, 1)
                probs = probs.cpu().detach().numpy()
                predictions += probs.tolist()
        return predictions

    def result_zero_cover(self, predictions):
        result_csv_df = pd.read_csv(self.submission_path)
        zero_csv1_df = pd.read_csv(self.zero_csv1_path)
        zero_csv2_df = pd.read_csv(self.zero_csv2_path)
        
        result_csv_df.iloc[:, 1:] = predictions
        
        indices1 = zero_csv1_df[(zero_csv1_df['fake'] == 0) & (zero_csv1_df['real'] == 0)].index
        indices2 = zero_csv2_df[(zero_csv2_df['fake'] == 0) & (zero_csv2_df['real'] == 0)].index
        
        and_idx = list(set(indices1).union(set(indices2)))
        result_csv_df.loc[and_idx, ['fake', 'real']] = 0.0

        result_csv_df.to_csv(self.save_path, index=False)
        return result_csv_df
