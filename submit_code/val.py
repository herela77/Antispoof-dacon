from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.calibration import calibration_curve

def validation(model, criterion, val_loader):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []

    best_val_score = float('inf')
    best_model = None
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(tqdm(iter(val_loader), desc="Validation")):
            features = features.unsqueeze(1)
            features = features.float().cuda()
            labels = labels.float().cuda()

            features = features.squeeze(1)
            outputs = model(features)

            loss = criterion(outputs, labels)
             
            probs = torch.sigmoid(outputs)

            val_loss.append(loss.item())

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        _val_loss = np.mean(val_loss)

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)

        combined_score = auc_brier_ece(all_labels, all_probs)
        
    print(f'Val Loss : [{_val_loss:.5f}] Combined Score : [{combined_score:.5f}]')
    
    
    if best_val_score > combined_score:
        best_val_score = combined_score
        best_model = model
    
    return best_model, best_val_score

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

