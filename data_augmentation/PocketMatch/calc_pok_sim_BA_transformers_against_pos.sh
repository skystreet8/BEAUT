#!/bin/bash
for o in A_muc B_Ang B_Dor C_You E_Rec R_Lac V_Vad
do
        echo "Processing $o, default pockets"
        ./Step3-PM_typeB "cabbage_files/${o}_filtered_pockets.cabbage" "cabbage_files/pos_v3_structs_high_index_sub_pockets.cabbage"
        mv PocketMatch_score.txt "results/${o}_default_against_pos.txt"
done
for o in A_muc B_Ang B_Dor C_You E_Rec R_Lac V_Vad
do
        echo "Processing $o, rescue pockets"
        ./Step3-PM_typeB "cabbage_files/${o}_rescue_filtered_pockets.cabbage" "cabbage_files/pos_v3_structs_high_index_sub_pockets.cabbage"
        mv PocketMatch_score.txt "results/${o}_rescue_against_pos.txt"
done
