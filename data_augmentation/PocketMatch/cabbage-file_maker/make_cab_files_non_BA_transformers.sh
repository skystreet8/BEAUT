#!/bin/bash
for o in A_muc B_Ang B_Dor C_You E_Rec R_Lac V_Vad
do
        bash Step0-cabbage.sh "non_BA_transformer_pockets/high_plddt_pockets_filtered/${o}"
        mv outfile.cabbage "../cabbage_files/${o}_filtered_pockets.cabbage"
done
for o in A_muc B_Ang B_Dor C_You E_Rec R_Lac V_Vad
do
        bash Step0-cabbage.sh "non_BA_transformer_pockets/high_plddt_pockets_rescue_filtered/${o}"
        mv outfile.cabbage "../cabbage_files/${o}_rescue_filtered_pockets.cabbage"
done
