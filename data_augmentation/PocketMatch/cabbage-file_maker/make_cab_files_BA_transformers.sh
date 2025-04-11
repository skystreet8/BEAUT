#!/bin/bash
for o in B_Ado B_Xyl C_Com C_M62_1 H_Fil R_Gna S_Inf
do
        bash Step0-cabbage.sh "BA_transformer_pockets/high_plddt_pockets_filtered/${o}"
        mv outfile.cabbage "../cabbage_files/${o}_filtered_pockets.cabbage"
done
for o in B_Ado B_Xyl C_Com C_M62_1 H_Fil R_Gna S_Inf
do
        bash Step0-cabbage.sh "BA_transformer_pockets/high_plddt_pockets_rescue_filtered/${o}"
        mv outfile.cabbage "../cabbage_files/${o}_rescue_filtered_pockets.cabbage"
done