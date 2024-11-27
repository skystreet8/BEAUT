import random
import pandas as pd
from utils import ReadFastaFile


NUM_FOLDS = 5
TEST_RATIO = 0.1
pos_headers, _ = ReadFastaFile('../data/positive_seqs_v2.fasta')
neg_headers, _ = ReadFastaFile('../data/negative_seqs_v2_wosimseqs.fasta')
NUM_NEGATIVE_SAMPLES = len(neg_headers) // 10
print(NUM_NEGATIVE_SAMPLES)

# Draw 10% random negative samples
neg_sample_headers = random.sample(neg_headers, NUM_NEGATIVE_SAMPLES)
data = []
for k in pos_headers:
    data.append([k, 1])
for k in neg_sample_headers:
    data.append([k, 0])
additional_test_headers = list(set(neg_headers) - set(neg_sample_headers))
for k in additional_test_headers:
    data.append([k, 0])
df = pd.DataFrame(data=data, columns=['header', 'label'])
pos_indexes = df.query('label == 1').index.values.tolist()
neg_indexes = df.query('label == 0').iloc[:len(neg_sample_headers)].index.values.tolist()
random.shuffle(pos_indexes)
random.shuffle(neg_indexes)
num_pos_seqs = len(pos_indexes)
num_neg_seqs = len(neg_indexes)
num_pos_test = int(num_pos_seqs * TEST_RATIO)
num_neg_test = 0
test_pos_indexes = pos_indexes[:num_pos_test]

test_indexes = test_pos_indexes + [i for i in range(len(pos_headers) + len(neg_sample_headers), len(df))]
pos_increment = (num_pos_seqs - len(test_pos_indexes)) // NUM_FOLDS
neg_increment = num_neg_seqs // NUM_FOLDS
print(pos_increment)
print(neg_increment)
pos_fold_indexes = [pos_indexes[num_pos_test + i * pos_increment:num_pos_test + (i + 1) * pos_increment]
                    for i in range(NUM_FOLDS - 1)]
pos_fold_indexes.append(pos_indexes[num_pos_test + (NUM_FOLDS - 1) * pos_increment:])
neg_fold_indexes = [neg_indexes[num_neg_test + i * neg_increment:num_neg_test + (i + 1) * neg_increment]
                    for i in range(NUM_FOLDS - 1)]
neg_fold_indexes.append(neg_indexes[num_neg_test + (NUM_FOLDS - 1) * neg_increment:])
fold_indexes = [t1 + t2 for t1, t2 in zip(pos_fold_indexes, neg_fold_indexes)]
fold_datasets = [['' for _ in range(len(df))] for _ in range(NUM_FOLDS)]
for i in range(NUM_FOLDS):
    for j in test_indexes:
        fold_datasets[i][j] = 'test'
    for j in fold_indexes[i]:
        fold_datasets[i][j] = 'val'
    for n in range(NUM_FOLDS):
        if n == i:
            continue
        for j in fold_indexes[n]:
            fold_datasets[i][j] = 'train'
for i in range(NUM_FOLDS):
    key = f'dataset_fold_{i + 1}'
    df[key] = fold_datasets[i]
for t in df.itertuples():
    temp = [t[i] for i in range(3, 3 + NUM_FOLDS)]
    assert temp.count('test') == NUM_FOLDS or (temp.count('val') == 1 and temp.count('train') == NUM_FOLDS - 1)
df.to_csv(f'../data/sequence_dataset_v2.csv', index=False)
