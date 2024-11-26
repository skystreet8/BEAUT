#!/bin/bash
# Provide the short organism name as the first argument
DIRECTORY="./data/BA_transformers/high_plddt_structs/$1_pdbs"

for FILE in "$DIRECTORY"/*
do
  if [ -f "$FILE" ]; then
    echo "Processing $FILE"
    ./cavity -r "$FILE" -o "./data/BA_transformers/high_plddt_pockets_rescue/$1" --rescue
  fi
done
