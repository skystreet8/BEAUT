from argparse import ArgumentParser
import torch
from models import DNNPredictor
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def predict(model, input_repr, threshold=0.5, return_prob=False):
    model.eval()
    with torch.no_grad():
        input_repr = input_repr.to(DEVICE)
        out = model(input_repr)
        probs = torch.softmax(out, dim=0)
        probs = probs.cpu().numpy()
        if return_prob:
            return list(probs)
        else:
            return int(probs[0] < probs[1])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--fasta', type=str, required=True, help='Name of the FASTA file')
    parser.add_argument('-th', '--thresh', type=float, default=0.5)
    args = parser.parse_args().__dict__
    seq_repr = torch.load(f'../data/case_embeddings/{args["fasta"]}.pt')
    seq_repr = seq_repr['mean_representations'][33]
    seq_repr = seq_repr.to(DEVICE)
    model = DNNPredictor(1280, [256, 32])
    model.load_state_dict(torch.load('../models/BEAUT_aug.pth')['model_state_dict'])
    model.to(DEVICE)
    prob = predict(model, seq_repr, threshold=args['thresh'], return_prob=True)
    print(round(prob[1], 4))
