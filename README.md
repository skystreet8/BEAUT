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
- Install ESM-2 following the README in the `esm` folder. 
## Training the Base & Aug model
### Step 1
Run `cd scripts`.

Run `python generate_datasets_base.py` and `python generate_datasets_aug.py`
to generate your data sets for model training. This first samples 10% (20% for
Aug model) positive sequences and puts them in the test set, then samples 10% random 
negative sequences and combine them with the remaining positive sequences to build the
training sets and validation sets for 5-fold internal cross-validation, and
puts the remaining 90% negative samples in the test set.

The data sets used to train
our own models are provided in the `data` folder. `sequence_dataset_v2.csv`
is for the Base model and `sequence_dataset_v2_substrate_pocket_aug_train_only.csv`
is for the Aug model. 
### Step 2
Run `python get_esm_reprs.py -m <model>` to get ESM-2 representations of
positive & negative sequences for Base & Aug model. `<model>` should be
either `base` or `aug`.
### Step 3
Run `python Train.py` to train the Base model. Run `python Train.py --aug`
to train the Aug model.

We provided the trained models in the `models` folder.

Run `python eval_metrics.py` or `python eval_metrics.py --aug` to
calculate the evaluation metrics for the Base or the Aug model, respectively.
The results will be saved at `../data/`.
We used the models with the best AUPR values in our study. The models were
copied and renamed to `BEAUT_base.pth` and `BEAUT_aug.pth`.
## Model usage
You can use `test_case.py` to test a single protein sequence.

For bulk predictions, use `test_bulk.py`. They have the same usage.

Usage: `python test_case.py -f <FASTA file name>`. Add `--aug` 
for using the Aug model. The Base model is default.

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
Run `python test_bulk.py -f PRJNA28331` and `python test_bulk.py -f PRJNA28331 --aug`
to get the prediction scores from the Base & the Aug model.

The results are saved at `../data/` by default. You need to make two
separate directories, `../data/PRJNA28331_base/` and `../data/PRJNA28331_aug/`
and place the results from the Base or the Aug model under the corresponding
directory. Following calculations assume you have done this and save
their results under the two directories. 
### Step 2
Run `python process_bulk_predictions.py -m <model>` to postprocess
the screening results. Sequences whose scores are above the threshold
are selected and are assigned with organisms.
### Step 3
Use EggNOG-mapper to get function annotations for the positive sequences.
We provided the annotation results in the corresponding folders.
Run `filter_non_enzymes_2.py -m <model>` to filter out non-enzymes
from previous positive sequences.

The annotation file is the `*.emapper.annotation` file produced by
EggNOG-mapper. We use `PRJNA28331_base.emapper.annotation` as the name
for the annotation file of the sequences predicted by the Base model.
For the filenames of annotations for the sequences predicted by the Aug model,
each patch (maximum 100,000 sequences) of annotation is named
`pt*.emapper.annotations`. Make sure you rename these files properly
and place them with the above prediction results under the same folders.
### Step 4
After sending the previous filtered sequences to CLEAN for EC prediction,
run `python add_clean_predictions.py -m <model>` to get the final results.

The predictions from CLEAN are precomputed and provided.
### Step 5
Run `python organism_stat.py -m <model>` to analyze the number of positive sequences
per organism and the number of positive sequences under each EC category per organism.

Run `python ec_stat.py -m <model>` to analyze the total number of positive sequences
under each EC category. KEGG descriptions for every EC category is provided. 
### Step 6
After sending the filtered sequences to EFI-EST and downloading the clustered
sequence network, run `python process_xgmml_graph.py -m <model>` to extract clusters.
Run `python process_ssn_clusters.py -m <model>` to analyze
the EC constitution for each cluster and assign cluster indexes to sequences.

The network file is precomputed and provided. This step requires at least 64 GB memory
and can take a long time to run.