# BEAUT
Code for "Identification of novel gut microbial bile acid metabolic enzymes
via a large-scale Al-assisted pipeline".

The necessary data for running and reproducing BEAUT are deposited at Zenodo.
Download these files from [here](https://zenodo.org/records/14233530?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjM4ZjZlNWVhLTBlOTctNDE2Yi04MjZiLTBjZGMwOTI4YjRiYyIsImRhdGEiOnt9LCJyYW5kb20iOiI5ODg4OGJiMzAyODEyZjE4YjAyZDA0N2M3ZWY1ZDYxOSJ9.FBmwdQury2JgkTvqs2h_TxGV4MTWB-23RmsisTbLeIDCMygVEHpSyCuGzeQdzkQXDgf6i2Ebg66-bdhDzx2U0g)
and follow the instructions on Zenodo to place 
the data folders under proper locations.
## Environment configuration
Requirements:

- Python 3.9
- PyTorch 1.13.1
- nltk
- pandas
- networkx
- scikit-learn
- tqdm
- biotite
- fair-esm >= 2
- DIAMOND 2.1.9.163
## Training the Aug model
`cd scripts`
### Step 1
This step removes redundant sequences at 90% sequence identity. First you should use DIAMOND to calculate the pairwise
sequence idenitites of the 469 primary positive sequences. Place your DIAMOND executive under `scripts` folder and run

`./diamond makedb --in ../data/positive_seqs_v3.fasta -d ../data/pos_seqs_v3`

`./diamond blastp -q ../data/positive_seqs_v3.fasta -d ../data/pos_seqs_v3 -o ../data/pos_seqs_v3_self_blast.tsv 
--ultra-sensitive -k 0`

Then run `python select_unique_seqs.py -f pos_seqs_v3`. This first processes the 469 primary positive sequences
(`../data/positive_seqs_v3.fasta`) and produces 151 non-redundant primary positive sequences with maximum pairwise
identity &#60; 90% (`../data/positive_seqs_v3_unique.fasta`). Also, these sequences are merged with the augmentation
sequences to produce 2392 augmented positive sequences in total (`../data/positive_seqs_v3_substrate_pocket_sim_aug_v3.fasta`).
After calculating pairwise sequence identities of these augmented positive sequences with DIAMOND, 

`./diamond makedb --in ../data/positive_seqs_v3_substrate_pocket_sim_aug_v3.fasta -d ../data/pos_seqs_v3_sub_pok_sim_aug_v3`

`./diamond blastp -q ../data/positive_seqs_v3_substrate_pocket_sim_aug_v3.fasta -d ../data/pos_seqs_v3_sub_pok_sim_aug_v3 -o ../data/pos_seqs_v3_sub_pok_sim_aug_v3_self_blast.tsv --ultra-sensitive -k 0`

run `python select_unique_seqs.py -f pos_seqs_v3_sub_pok_sim_aug_v3` to remove redundant sequences from the 2392 augmented
positive sequences (`../data/positive_seqs_v3_substrate_pocket_sim_aug_v3_unique.fasta`).
This should give 2383 sequences in total which will be used to generate the data sets.

As this step could be random, the files generated in developing the model are provided in the data archive. You can
directly use the archived files to reproduce our results. 
### Step 2
Use DIAMOND to calculate the pairwise sequence identities for the 2383 augmented positive sequences:

`./diamond makedb --in ../data/positive_seqs_v3_substrate_pocket_sim_aug_v3_unique.fasta -d ../data/pos_seqs_v3_sub_pok_sim_aug_v3_uniq`

`./diamond blastp -q ../data/positive_seqs_v3_substrate_pocket_sim_aug_v3_unique.fasta -d ../data/pos_seqs_v3_sub_pok_sim_aug_v3_uniq -o ../data/pos_seqs_v3_sub_pok_sim_aug_v3_uniq_self_blast.tsv --ultra-sensitive -k 0`

Run `python generate_datasets_aug.py`
to generate your data sets for model training. This first clusters the augmented positive samples according to their
pairwise sequence identities with a threshold of 30% and randomly samples a number of clusters to place them
in the test set. The total number of sequences from the clusters is 10% of the number of postive samples. Note
that if a cluster fully consists of primary positive samples, then this cluster will be sampled prior to other clusters.
The remaining positive samples are used to build the 5-fold cross-validation data sets. The sequence lengths of these
samples are calculated first and then the samples are divided into groups of different length ranges. For each group,
negative samples 5 times the size of the group whose lengths are within the same range are sampled and added to the
group. In each fold of cross-validation, we selected 1/5 samples from each group to form the validation set and the
remaining samples form the training set. 

Negative samples that are not sampled for training & validation are all placed
in the test set. As this would cause a huge data imbalance in the test set compared to the validation set and might
affect model evaluation, we separately build 5 additional test sets with the same positive test samples and the same
positive sample ratio to the validation set which is 1:5. The negative samples in each test set are randomly sampled
and have no overlap. Evaluation will be performed on both the full test set and the 5 re-balanced test sets.

The data sets used to train
our own models are provided in the `data` folder with the name `sequence_dataset_v3_substrate_pocket_aug.csv`.
### Step 3
Run `python Train.py` to train the Aug model. This step requires the sequence embedding data `../data/seq_embeddings_v3_substrate_pocket_aug.pt`
which covers all negative sequences and the 2383 augmented positive sequences. 

We provided the trained models in the `models` folder.

Run `python eval_metrics.py` to calculate the evaluation metrics for the Aug model.
The results will be saved at `../data/BEAUT_aug_eval_metrics.csv`.
We used the model with the best AUPR value in our study. The model was
copied and renamed `BEAUT_aug.pth`.

To evaluate the best model with balanced test sets(as described in Step 1), run `python generate_balanced_test_sets.py`
and then run `python eval_metrics_balanced.py`. The results on 5 balanced test sets are saved at `../data/BEAUT_aug_eval_metrics_balanced.csv`.
## Model usage
You can use `test_case.py` to test a single protein sequence.

For bulk predictions, use `test_bulk.py`. They have the same usage.

Usage: `python test_case.py -f <FASTA file name>`.

Before prediction, you need to compute the ESM-2 representations.

First, `cd ../esm`. If you only have one sequence, run `bash esm-extract.sh <your FASTA file name>.fasta ../data/case_embeddings/`.
Then `cd ../scripts` and immediately run `python convert_embedding_chunks.py -f <your FASTA file name>`. If not the next run of ESM-2
would overwrite the previous output. `<your FASTA file name>` should not include the `.fasta` postfix.

If you have multiple sequences, run `bash esm-extract.sh <your FASTA file name>.fasta ../data/`. Then `cd ../scripts` and 
run `python convert_embedding_chunks.py -f <your FASTA file name> --multiple`.

`test_case.py` will directly print out the predicted probability.
`test_bulk.py` will save the results as *.pkl files in the `data` folder.
## Screening the PRJNA28331 dataset
### Step 1
Run `python test_bulk.py -f PRJNA28331` to get the prediction scores from the Aug model.

The results are saved at `../data/` by default. You need to make a directory `../data/PRJNA28331_aug/`
and place the result from the Aug model under the directory.
Following calculations assume you have done this and save their results under the directory. 
### Step 2
Run `python process_bulk_predictions.py` to postprocess
the screening results. Sequences whose scores are above the threshold
are selected and are assigned with organisms.
### Step 3
Use EggNOG-mapper to get function annotations for the positive sequences.
We provided the annotation results in the corresponding folders.
Run `filter_non_enzymes_2.py` to filter out non-enzymes
from previous positive sequences.

Each patch (maximum 100,000 sequences) of annotation is named
`pt*.emapper.annotations`. Make sure you rename these files properly
and place them under the same folder with the above prediction results.
### Step 4
After sending the previous filtered sequences to CLEAN for EC prediction,
run `python add_clean_predictions.py` to get the final results.

The predictions from CLEAN are precomputed and provided.
### Step 5
Run `python organism_stat.py` to analyze the number of positive sequences
per organism and the number of positive sequences under each EC category per organism.

Run `python ec_stat.py` to analyze the total number of positive sequences
under each EC category. KEGG descriptions for every EC category is provided. 
### Step 6
After sending the filtered sequences to EFI-EST and downloading the clustered
sequence network, run `python process_xgmml_graph.py` to extract clusters.
Run `python process_ssn_clusters.py` to analyze
the EC constitution for each cluster and assign cluster indexes to sequences.

The network file is precomputed and provided. This step requires at least 64 GB memory
and can take a long time to run.