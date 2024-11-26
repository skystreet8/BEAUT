#!/bin/bash
for o in B_Ado B_Xyl C_Com C_M62_1 H_Fil R_Gna S_Inf
do
        echo "Processing $o, default pockets"
        ./Step3-PM_typeB "cabbage_files/${o}_filtered_pockets.cabbage" "cabbage_files/pos_v2_structs_high_index_sub_pockets.cabbage"
        mv PocketMatch_score.txt "results/${o}_default_against_pos.txt"
done
for o in B_Ado B_Xyl C_Com C_M62_1 H_Fil R_Gna S_Inf
do
        echo "Processing $o, rescue pockets"
        ./Step3-PM_typeB "cabbage_files/${o}_rescue_filtered_pockets.cabbage" "cabbage_files/pos_v2_structs_high_index_sub_pockets.cabbage"
        mv PocketMatch_score.txt "results/${o}_rescue_against_pos.txt"
done
