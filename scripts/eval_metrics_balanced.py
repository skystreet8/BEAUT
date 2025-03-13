import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from models import DNNPredictor
from dataset import SequenceTestDataset
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
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            reprs = batch_data
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
    return test_probs, predictions


if __name__ == '__main__':
    threshold = 0.5
    auprs = []
    f1_scores = []
    f2_scores = []
    mccs = []
    precs = []
    recs = []
    conf_mats = []
    model = DNNPredictor(1280, [256, 32])
    results = {}
    for i in tqdm(range(1, 6), total=5):
        test_df = pd.read_csv(f'../data/test_set_sample_{i}.csv')
        header2label = {t[1]: t[2] for t in test_df.itertuples()}
        test_dataset = SequenceTestDataset(f'../data/test_set_sample_{i}_embeddings.pt')
        test_dataloader = DataLoader(test_dataset, batch_size=128)
        model.load_state_dict(torch.load(f'../models/BEAUT_aug.pth')['model_state_dict'])
        model.to(DEVICE)
        test_probs, test_predictions = predict(model, test_dataloader, threshold=threshold)
        test_headers = test_dataset.headers
        test_labels = [header2label[h] for h in test_headers]

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
    df = pd.DataFrame(data=None)
    df.index = [f'test_set_sample_{i}' for i in range(1, 6)]
    df = df.assign(AUPR=auprs, Recall=recs, Precision=precs, F1_score=f1_scores, F2_score=f2_scores,
                   MCC=mccs, Confusion_matrix=conf_mats)
    df.to_csv('../data/BEAUT_aug_eval_metrics_balanced.csv', index=True)
