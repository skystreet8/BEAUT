#!/bin/bash

DIRECTORY="./data/positive_seqs_v2_pdbs_high_plddt"

for FILE in "$DIRECTORY"/*
do
  if [ -f "$FILE" ]; then
    echo "Processing $FILE"
    ./cavity -r "$FILE" -o ./pos_v2_structs_high_plddt_pockets_rescue/ --rescue
  fi
done
