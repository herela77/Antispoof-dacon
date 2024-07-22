from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.calibration import calibration_curve
import torch

def train_unlabel(model, optimizer, criterion, train_loader, epoch=9999):
    model.train()
    train_loss, all_probs = [], []
    tqdm_train = tqdm(train_loader)
    up_thres = 0.9
    down_thres = 0.1
    for _, features in enumerate(tqdm_train):
        features = features.float().cuda()
        optimizer.zero_grad()
        outputs = model(features)
    
        with torch.no_grad():
            pseudo_outputs = model(features)
            pseudo_probs = torch.sigmoid(pseudo_outputs)
            pseudo_labels = torch.zeros_like(pseudo_probs)
            pseudo_labels[pseudo_probs >= up_thres] = 1
            pseudo_labels[pseudo_probs <= down_thres] = 0
            mask = (pseudo_probs >= up_thres) | (pseudo_probs <= down_thres)
        
        filtered_outputs = outputs[mask]
        filtered_pseudo_labels = pseudo_labels[mask]

        loss = criterion(filtered_outputs, filtered_pseudo_labels)


        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        probs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
        all_probs.append(probs.detach().cpu().numpy())

        _train_loss = np.mean(train_loss)
        tqdm_train.set_description('Train Epoch: {}, Average loss: {:.6f}'.format(epoch, _train_loss))
        
    all_probs = np.concatenate(all_probs, axis=0)
    
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
