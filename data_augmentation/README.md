# Producing augmentation samples based on substrate pocket similarity
First, `cd ./data_augmentation/scripts`.
## 1. Processing genome sequences from selected organisms
First, run `python filter_non_enzymes_1.py` to remove non-enzymes and possible enzymes involving macromolecules like
proteins, nucleic acids as substrates. This would produce 7 `*_filtered_by_annotation.fasta` files and 
7 `*_filtered_with_annotations.csv` files. The fasta files are sequences contributed by each organism and the CSV files
are the corresponding function annotations from EggNOG-mapper.

Then, we remove possible negative sequences using BLAST with the genome sequences as query and our negative sequences as target.
Run DIAMOND with the following commands:

`ln -s ../../scripts/diamond ./diamond`

`./diamond makedb --in ../../data/negative_seqs_v2.fasta -d ../data/neg_seqs_v2`

`./diamond blastp -q ../data/BA_transformers/<organism>_filtered_by_annotation.fasta -d ../data/neg_seqs_v2 
-o ../data/BA_transformers/<organism>_against_neg_seqs_v2_blast.tsv --id 50`

The available organisms are `B_Ado, B_Xyl, C_Com, C_M62_1, H_Fil, R_Gna, S_Inf`.

Then run `python remove_possible_neg_seqs.py` to remove sequences that have at least one hit
with identity &#8805; 50 and bitscore &#8805; 50 to the negative sequences. 
## 2. Filter PDB files by pLDDT
Run `python get_high_plddt_structs.py -o <organism>` to filter PDB files of the proteins from selected sequences according to
pLDDT. The available organisms are `B_Ado, B_Xyl, C_Com, C_M62_1, H_Fil, R_Gna, S_Inf`.
## 3. Extract pockets using Cavity
Run `cd ..`.

Place your Cavity program or create a soft link at `./data_augmentation`.
Run `bash extract_pockets_high_plddt_BA_transformer_structs.sh <organism>`
and `bash extract_pockets_high_plddt_BA_transformer_structs_rescue.sh <organism>`
to extract pockets from high pLDDT structures
of the proteins from selected organisms.
The available organisms are the same as above.

As this step requires very long time, precomputed pockets(gzipped files) are provided 
in the `data/BA_transformers/` folder. 
## 4. Filter out the query pockets from all pockets extracted from protein structures of the 7 selected organisms
Run `python filter_pockets.py -o <organism>` and `python filter_pockets.py -o <organism> -r`.
This would extract query pockets with volumes of 1000-5500 &#x00C5;^3 and pocket indexes &#x2265; 0.7.
Pockets are converted to PocketMatch acceptable format. 

The reference pockets from the primary positive samples were prepared manually.
They are provided in the `../PocketMatch/cabbage-file_maker/` folder
as `pos_v3_structs_high_index_sub_pockets.tar.gz`.
## 5. Run PocketMatch to get pocket similarity results
First, make two folders `cabbage_files` and `results` under the `../PocketMatch` folder.
Make a folder `BA_transformer_pockets` under the `../PocketMatch/cabbage-file_maker/` folder.
Then change to the `scripts` folder as your working directory and continue.
### 5.1 Make cabbage files
Copy the folders containing the query pockets to `../PocketMatch/cabbage-file_maker/BA_transformer_pockets/`.

`cp -r "../data/BA_transformers/high_plddt_pockets_filtered/" ../PocketMatch/cabbage-file_maker/BA_transformer_pockets/`

`cp -r ../data/BA_transformers/high_plddt_pockets_rescue_filtered/ ../PocketMatch/cabbage-file_maker/BA_transformer_pockets/`

Then `cd ../PocketMatch/cabbage-file_maker`.
Run `bash ./Step0-cabbage.sh pos_v3_structs_high_index_sub_pockets/` to make the
cabbage file for the reference pockets. Run `mv outfile.cabbage ../cabbage_files/pos_v3_structs_high_index_sub_pockets.cabbage`
immediately. Otherwise, the next time you run `bash ./Step0-cabbage.sh` will 
overwrite `outfile.cabbage`.

Run `bash make_cab_files_BA_transformers.sh` to make the cabbage files for the
query pockets. This will produce 14 cabbage files in the `../cabbage_files/`
folder, two for one organism.

If many `Permission denied` appear as error messages, run the following 
commands under the `cabbage-file_maker` folder:

`chmod +x Step0-cabbage_core`

`chmod +x Step0-cabbage_decoder`

`chmod +x Step0-END-FILE`
### 5.2 Run similarity calculations
First, `cd ..`.

Run `chmod +x ./Step3-PM_typeB` to avoid permission errors.

Run `bash calc_pok_sim_BA_transformers_against_pos.sh`.
This will give 14 files containing the pairwise pocket similarity
between the query pockets and the reference pockets in the `results` folder.
## 6. Analyze pocket similarity and collect augmentation data
Run `python process_pocketmatch_outputs.py`.
This will produce the augmentation samples as `substrate_pocket_sim_aug_v3.fasta`.

This will produce two additional files, `BA_transformers_default_matched_pockets_against_pos.csv`
and `BA_transformers_rescue_matched_pockets_against_pos.csv` in the `../data/`folder.
They record pocket pairs with similarities &#x2265; 0.7. 