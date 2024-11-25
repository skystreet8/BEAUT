from argparse import ArgumentParser
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Subset, DataLoader
from models import DNNPredictor
from dataset import SequenceDataset
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef
logger = logging.getLogger('Test')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def predict(model, test_dataloader):
    model.eval()
    predictions = []
    test_labels = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            reprs, labels = batch_data
            labels = labels.numpy()
            labels = np.squeeze(labels)
            logger.debug(labels.shape)
            test_labels.extend(list(labels))
            reprs = reprs.to(DEVICE)
            out = model(reprs)
            probs = torch.sigmoid(out)
            probs = probs.cpu().numpy()
            probs = np.squeeze(probs)
            logger.debug(probs.shape)  # Should be (batch_size, 10117)
            predictions.extend(list(probs))
    return predictions, test_labels


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--aug', action='store_true', default=False)
    args = parser.parse_args()
    aug = args.aug
    if aug:
        threshold = 0.8
    else:
        threshold = 0.9
    auprs = []
    f1_scores = []
    mccs = []
    precisions = []
    model = DNNPredictor(1280, [256, 32])
    dataset = SequenceDataset(fold=1, aug=aug)
    test_dataset = Subset(dataset, indices=dataset.test_ids)
    test_dataloader = DataLoader(test_dataset, batch_size=128)
    for fold in tqdm(range(1, 6), total=5):
        dataset.set_fold(fold)
        if aug:
            model.load_state_dict(torch.load(f'../models/ba_pred_DNN_aug_fold_{fold}.pth')['model_state_dict'])
        else:
            model.load_state_dict(torch.load(
                f'../models/ba_pred_DNN_base_fold_{fold}.pth')['model_state_dict'])
        model.to(DEVICE)
        test_predictions, test_labels = predict(model, test_dataloader)
        test_predictions = np.array(test_predictions)
        test_labels = np.array(test_labels)
        test_pred_labels = np.where(test_predictions > threshold, 1, 0)
        aupr = round(average_precision_score(test_labels, test_predictions), 4)
        auprs.append(aupr)
        f1score = round(f1_score(test_labels, test_pred_labels), 4)
        f1_scores.append(f1score)
        mcc = round(matthews_corrcoef(test_labels, test_pred_labels), 4)
        mccs.append(mcc)

    data = list(range(1, 6))
    df = pd.DataFrame(data, columns=['fold'])
    df = df.assign(AUPR=auprs, F1_score=f1_scores, MCC=mccs)
    if aug:
        df = df.assign(precision=precisions)
        df.to_csv('../data/ba_pred_DNN_aug_eval_metrics.csv', index=False)
    else:
        df.to_csv('../data/ba_pred_DNN_base_eval_metrics.csv', index=False)
