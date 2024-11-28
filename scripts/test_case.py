from argparse import ArgumentParser
import torch
from models import DNNPredictor
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def predict(model, input_repr, threshold=0.5, return_prob=False):
    model.eval()
    with torch.no_grad():
        input_repr = input_repr.to(DEVICE)
        out = model(input_repr)
        prob = torch.sigmoid(out)
        if return_prob:
            return prob
        if prob > threshold:
            return 1
        else:
            return 0


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--fasta', type=str, required=True, help='Name of the FASTA file')
    parser.add_argument('-th', '--thresh', type=float, default=0.5)
    parser.add_argument('--aug', action='store_true', default=False)
    args = parser.parse_args().__dict__
    seq_repr = torch.load(f'../data/case_embeddings/{args["fasta"]}.pt')
    seq_repr = seq_repr['mean_representations'][33]
    seq_repr = seq_repr.to(DEVICE)
    model = DNNPredictor(1280, [256, 32])
    if args['aug']:
        model.load_state_dict(torch.load('../models/BEAUT_aug.pth')['model_state_dict'])
    else:
        model.load_state_dict(torch.load('../models/BEAUT_base.pth')['model_state_dict'])
    model.to(DEVICE)
    prob = predict(model, seq_repr, threshold=args['thresh'], return_prob=True)
    prob = prob.detach().cpu().item()
    print(round(prob, 4))
