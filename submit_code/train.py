from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.calibration import calibration_curve
import torch
from util import mixup, cutmix

def train(model, optimizer, criterion, train_loader, scheduler, epoch=9999):
    model.train()
    train_loss, all_labels, all_probs = [], [], []
    tqdm_train = tqdm(train_loader)

    for _, (features, labels) in enumerate(tqdm_train):
        features = features.unsqueeze(1)
        features = features.float().cuda()
        labels = labels.float().cuda()
        rand_num = np.random.rand()
        if rand_num < 0.5:
            features, label_mix = cutmix(features, labels, 0.6)

        elif rand_num >= 0.5:
            features, label_mix = mixup(features, labels, 0.7)
        
        model.zero_grad()
        
        features = features.squeeze(1) #ast
        outputs = model(features)
        label_1, label_2, lam = label_mix

        loss = lam * criterion(outputs, label_1) + (1- lam) * criterion(outputs, label_2)

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss.append(loss.item())

        probs = torch.sigmoid(outputs)
        all_labels.append(labels.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

        _train_loss = np.mean(train_loss)
        tqdm_train.set_description('Train Epoch: {}, Average loss: {:.6f}'.format(epoch, _train_loss))
        
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    
    combined_score = auc_brier_ece(all_labels, all_probs)
    print("train_score : ", combined_score)
    return _train_loss



def expected_calibration_error(y_true, y_prob, n_bins=10):
    ece_list = []
    for i in range(y_true.shape[1]):
        prob_true, prob_pred = calibration_curve(y_true[:, i], y_prob[:, i], n_bins=n_bins, strategy='uniform')
        bin_totals = np.histogram(y_prob[:, i], bins=np.linspace(0, 1, n_bins + 1), density=False)[0]
        non_empty_bins = bin_totals > 0
        bin_weights = bin_totals / len(y_prob)
        bin_weights = bin_weights[non_empty_bins]
        prob_true = prob_true[:len(bin_weights)]
        prob_pred = prob_pred[:len(bin_weights)]
        ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
        ece_list.append(ece)
    return np.mean(ece_list)
    
def auc_brier_ece(y_true, y_prob):
    auc_scores = []
    brier_scores = []
    ece_scores = []

    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        auc_scores.append(auc)
        
        brier = mean_squared_error(y_true[:, i], y_prob[:, i])
        brier_scores.append(brier)
        
        ece = expected_calibration_error(y_true, y_prob)
        ece_scores.append(ece)
    
    mean_auc = np.mean(auc_scores)
    mean_brier = np.mean(brier_scores)
    mean_ece = np.mean(ece_scores)
    
    combined_score = 0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece
    
    return combined_score
