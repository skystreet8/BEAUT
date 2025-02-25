import pickle
from argparse import ArgumentParser
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Subset, DataLoader
from models import DNNPredictor
from dataset import SequenceDataset
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score,\
    confusion_matrix, precision_recall_curve, auc
logger = logging.getLogger('Test')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def predict(model, test_dataloader):
    model.eval()
    predictions = []
    test_labels = []
    test_headers = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            reprs, labels, headers = batch_data
            labels = labels.numpy()
            labels = np.squeeze(labels)
            logger.debug(labels.shape)
            test_labels.extend(list(labels))
            test_headers.extend(list(headers))
            reprs = reprs.to(DEVICE)
            out = model(reprs)
            probs = torch.sigmoid(out)
            probs = probs.cpu().numpy()
            probs = np.squeeze(probs)
            logger.debug(probs.shape)  # Should be (batch_size, 10117)
            predictions.extend(list(probs))
    return predictions, test_labels, test_headers


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--aug', action='store_true', default=False)
    args = parser.parse_args()
    aug = args.aug
    if aug:
        threshold = 0.85
    else:
        threshold = 0.9
    auprs = []
    f1_scores = []
    mccs = []
    conf_mats = []
    model = DNNPredictor(1280, [256, 32])
    dataset = SequenceDataset(fold=1, aug=aug)
    test_dataset = Subset(dataset, indices=dataset.test_ids)
    test_dataloader = DataLoader(test_dataset, batch_size=128)
    for fold in tqdm(range(1, 6), total=5):
        dataset.set_fold(fold)
        if aug:
            model.load_state_dict(torch.load(f'../models/BEAUT_aug_fold_{fold}.pth')['model_state_dict'])
        else:
            model.load_state_dict(torch.load(
                f'../models/BEAUT_base_fold_{fold}.pth')['model_state_dict'])
        model.to(DEVICE)
        test_predictions, test_labels, test_headers = predict(model, test_dataloader)
        test_predictions = np.array(test_predictions)
        sort_indexes = np.argsort(test_predictions)[::-1]
        test_labels = np.array(test_labels)
        test_predictions = np.array([test_predictions[i] for i in sort_indexes])
        test_labels = np.array([test_labels[i] for i in sort_indexes])
        test_headers = [test_headers[i] for i in sort_indexes]
        print(f'Fold: {fold}')
        pos_label_ranks = list(np.argwhere(test_labels == 1).reshape(-1))
        pos_probs = [round(float(test_predictions[i]), 4) for i in pos_label_ranks]
        print(pos_label_ranks)
        print(pos_probs)
        print([test_headers[i] for i in pos_label_ranks])
        test_pred_labels = np.where(test_predictions > threshold, 1, 0)
        precisions, recalls, _ = precision_recall_curve(test_labels, test_predictions, pos_label=1)
        if fold == 1:
            pickle.dump({'precision': precisions, 'recall': recalls}, open('../data/PR_values_fold_1.pkl', 'wb'))
        aupr = round(auc(recalls, precisions), 4)
        auprs.append(aupr)
        f1score = round(f1_score(test_labels, test_pred_labels), 4)
        f1_scores.append(f1score)
        mcc = round(matthews_corrcoef(test_labels, test_pred_labels), 4)
        mccs.append(mcc)
        conf_mat = confusion_matrix(test_labels, test_pred_labels)
        conf_mats.append(list(conf_mat))
    data = list(range(1, 6))
    df = pd.DataFrame(data, columns=['fold'])
    df = df.assign(AUPR=auprs, F1_score=f1_scores, MCC=mccs, conf_mat=conf_mats)
    if aug:
        df.to_csv('../data/BEAUT_aug_eval_metrics.csv', index=False)
    else:
        df.to_csv('../data/BEAUT_base_eval_metrics.csv', index=False)
