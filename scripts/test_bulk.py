import pickle
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import DNNPredictor
from dataset import SequenceTestDataset
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def predict(model, input_repr):
    model.eval()
    with torch.no_grad():
        input_repr = input_repr.to(DEVICE)
        out = model(input_repr)
        probs = torch.softmax(out, dim=1)
        probs = probs.cpu().numpy()
        return probs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--fasta', type=str, required=True, help='Path to the FASTA file')
    parser.add_argument('-th', '--thresh', type=float, default=0.9)
    parser.add_argument('--aug', action='store_true', default=False)
    args = parser.parse_args().__dict__
    test_dataset = SequenceTestDataset(f'../data/{args["fasta"]}_embeddings.pt')
    test_dataloader = DataLoader(test_dataset, batch_size=128)
    all_probs = []
    all_res = []
    model = DNNPredictor(1280, [256, 32])
    if args['aug']:
        model.load_state_dict(torch.load('../models/BEAUT_aug.pth')['model_state_dict'])
    else:
        model.load_state_dict(torch.load('../models/BEAUT_base.pth')['model_state_dict'])
    model.to(DEVICE)
    for batch_seq_reprs in tqdm(test_dataloader, total=len(test_dataloader)):
        batch_seq_reprs = batch_seq_reprs.to(DEVICE)
        prob = predict(model, batch_seq_reprs)
        batch_predictions = np.argmax(prob, axis=1)
        all_probs.extend(list(prob))
        all_res.extend(list(batch_predictions))
    print(all_res.count(1))
    result = {k: v for k, v in zip(test_dataset.headers, all_probs)}
    if args['aug']:
        with open(f'../data/{args["fasta"]}_results_BEAUT_aug.pkl', 'wb') as f:
            pickle.dump(result, f)
        f.close()
    else:
        with open(f'../data/{args["fasta"]}_results_BEAUT_base.pkl', 'wb') as f:
            pickle.dump(result, f)
        f.close()
