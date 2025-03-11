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
    confusion_matrix, fbeta_score
logger = logging.getLogger('Test')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def predict(model, test_dataloader, threshold=0.5):
    model.eval()
    test_probs = []
    predictions = []
    test_labels = []
    test_headers = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            reprs, labels, headers = batch_data
            labels = labels.numpy()
            labels = labels.squeeze()
            logger.debug(labels.shape)
            test_labels.extend(list(labels))
            test_headers.extend(list(headers))
            reprs = reprs.to(DEVICE)
            out = model(reprs)
            probs = torch.softmax(out, dim=1)
            probs = probs.cpu().numpy()
            probs = list(np.squeeze(probs))
            test_probs.extend(probs)
            batch_predictions = []
            for t in probs:
                if t[0] - t[1] < 1 - threshold * 2:
                    batch_predictions.append(1)
                else:
                    batch_predictions.append(0)
            predictions.extend(list(batch_predictions))
    return test_probs, predictions, test_labels, test_headers


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--aug', action='store_true', default=False)
    args = parser.parse_args()
    aug = args.aug
    if aug:
        threshold = 0.5
    else:
        threshold = 0.9
    auprs = []
    f1_scores = []
    f2_scores = []
    mccs = []
    precs = []
    recs = []
    conf_mats = []
    model = DNNPredictor(1280, [256, 32])
    dataset = SequenceDataset(fold=1, aug=aug)
    test_dataset = Subset(dataset, indices=dataset.test_ids)
    test_dataloader = DataLoader(test_dataset, batch_size=128)
    results = {}
    for fold in tqdm(range(1, 6), total=5):
        dataset.set_fold(fold)
        if aug:
            model.load_state_dict(torch.load(f'../models/BEAUT_aug_fold_{fold}.pth')['model_state_dict'])
        else:
            model.load_state_dict(torch.load(
                f'../models/BEAUT_base_fold_{fold}.pth')['model_state_dict'])
        model.to(DEVICE)
        test_probs, test_predictions, test_labels, test_headers = predict(model, test_dataloader, threshold=threshold)
        results[fold] = [test_probs, test_labels, test_headers]

        pos_probs = [t[1] for t in test_probs]
        aupr = average_precision_score(test_labels, pos_probs)
        auprs.append(aupr)
        test_predictions = np.array(test_predictions)
        test_labels = np.array(test_labels)
        rec = round(recall_score(test_labels, test_predictions), 4)
        recs.append(rec)
        prec = round(precision_score(test_labels, test_predictions), 4)
        precs.append(prec)
        f1score = round(f1_score(test_labels, test_predictions), 4)
        f1_scores.append(f1score)
        f2score = round(fbeta_score(test_labels, test_predictions, beta=2), 4)
        f2_scores.append(f2score)
        mcc = round(matthews_corrcoef(test_labels, test_predictions), 4)
        mccs.append(mcc)
        conf_mat = confusion_matrix(test_labels, test_predictions)
        conf_mats.append(list(conf_mat))
    data = list(range(1, 6))
    df = pd.DataFrame(data, columns=['fold'])
    df = df.assign(AUPR=auprs, Recall=recs, Precision=precs, F1_score=f1_scores, F2_score=f2_scores,
                   MCC=mccs, Confusion_matrix=conf_mats)
    if aug:
        df.to_csv('../data/BEAUT_aug_eval_metrics.csv', index=False)
    else:
        df.to_csv('../data/BEAUT_base_eval_metrics.csv', index=False)
    pickle.dump(results, open('../data/BEAUT_aug_test_results.pkl', 'wb'))