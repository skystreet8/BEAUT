/*Step0-cabbage_core.h (header file):
This file contains structures and functions that facilitate easy IP/OP of modified cabbage files.
Also contains an implementation of Kali's binning algorithm.
It is dependent on the "Step0-PDBclass.h" object-file.
*/

/*STRUCTURES: Output cabbage files with 3 variables:
-<pairwise distance>
-Matched atoms in site-1
-Matched atoms in site-2
UNIONS used for easy IP/OP of ints, doubles as bit-strings.
*/


typedef union  {
double d;
char c[sizeof(double)];
} UNION_double;

typedef union  {
int i;
char c[sizeof(int)];
} UNION_int;

typedef struct  {
  UNION_double distance;
  UNION_int residue1;
  UNION_int residue2;
  } element;

//FUNCTION: Write object 'PDBclass' input-lines into structure 'element' output:
element make_element(PDBclass IP_line1, PDBclass IP_line2)
{
element elem;
elem.distance.d = sqrt( pow(IP_line1.C7_coords[0]-IP_line2.C7_coords[0],2)
                  +pow(IP_line1.C7_coords[1]-IP_line2.C7_coords[1],2)
                  +pow(IP_line1.C7_coords[2]-IP_line2.C7_coords[2],2)
                 );
//Insert 'chain' as 100000*chain_character:
elem.residue1.i = IP_line1.C6_residue +CHAIN_FACTOR*((int)IP_line1.C5_chain-'@');
elem.residue2.i = IP_line2.C6_residue +CHAIN_FACTOR*((int)IP_line2.C5_chain-'@');
return elem;
}

//FUNCTION: Assign bins (1-90) to various matches:
int binner(PDBclass IP_line1, PDBclass IP_line2)
{
/*Kali has a strange binning algorithm. I've tried to emulate it the best I can,
but I had to create primary, secondary, and final bins to do so:

PRIMARY BINS: They range from (0-4) for <residue groups> and (0-2) for <structural groups>.
There are 4 types of bins in all.
SECONDARY BINS: They range from (0-14) pairs of <residue groups> and (0-5) pairs of <structural groups>.
There are 2 types of bins in all.
FINAL BINS: They range from (0-90), and reflect all combinations of secondary bins.
There is only 1 type of final bin.

On a personal note, I don't think this binning system, or breaking <distance elements> into sorted-stacks, reflects biological, or even 3D-structural realities;
but hey, I'm just the high-performance guy.
*/
int i, j, k, l;
int RES_GROUPS=5, STR_GROUPS=3;
//'COMBO' variables denote total number of combinations for a given number of RES/STR groups.
//These numbers will be calculated later inside loops:
int RES_COMBOS=0, STR_COMBOS=0;

//PRIMARY BIN ASSIGNMENT:
int residue_group1, residue_group2, structural_group1, structural_group2;

//Assign <residue_groups> to amino acids:
//First assignment:
switch(IP_line1.C4_short)  {
  case 'G': residue_group1 = 0; break;
  case 'A': residue_group1 = 0; break;
  case 'V': residue_group1 = 0; break;
  case 'L': residue_group1 = 0; break;
  case 'I': residue_group1 = 0; break;
  case 'P': residue_group1 = 0; break;
  case 'M': residue_group1 = 0; break;

  case 'K': residue_group1 = 1; break;
  case 'R': residue_group1 = 1; break;
  case 'H': residue_group1 = 1; break;

  case 'D': residue_group1 = 2; break;
  case 'E': residue_group1 = 2; break;
  case 'Q': residue_group1 = 4; break; //UNDOCUMENTED change
  case 'N': residue_group1 = 4; break; //UNDOCUMENTED change

  case 'Y': residue_group1 = 3; break;
  case 'F': residue_group1 = 3; break;
  case 'W': residue_group1 = 3; break;

  case 'C': residue_group1 = 4; break;
  case 'S': residue_group1 = 4; break;
  case 'T': residue_group1 = 4; break;
  }
//Second assignment:
switch(IP_line2.C4_short)  {
  case 'G': residue_group2 = 0; break;
  case 'A': residue_group2 = 0; break;
  case 'V': residue_group2 = 0; break;
  case 'L': residue_group2 = 0; break;
  case 'I': residue_group2 = 0; break;
  case 'P': residue_group2 = 0; break;
  case 'M': residue_group2 = 0; break;

  case 'K': residue_group2 = 1; break;
  case 'R': residue_group2 = 1; break;
  case 'H': residue_group2 = 1; break;

  case 'D': residue_group2 = 2; break;
  case 'E': residue_group2 = 2; break;
  case 'Q': residue_group2 = 4; break; //UNDOCUMENTED change
  case 'N': residue_group2 = 4; break; //UNDOCUMENTED change

  case 'Y': residue_group2 = 3; break;
  case 'F': residue_group2 = 3; break;
  case 'W': residue_group2 = 3; break;

  case 'C': residue_group2 = 4; break;
  case 'S': residue_group2 = 4; break;
  case 'T': residue_group2 = 4; break;
  }

//Assign <structural_groups> to amino acids:
//First assignment:
if(strncmp(IP_line1.C3_element," CA ",4)==0) structural_group1 = 0;
if(strncmp(IP_line1.C3_element," CB ",4)==0) structural_group1 = 1;
if(strncmp(IP_line1.C3_element," CR ",4)==0) structural_group1 = 2;
//Second assignment:
if(strncmp(IP_line2.C3_element," CA ",4)==0) structural_group2 = 0;
if(strncmp(IP_line2.C3_element," CB ",4)==0) structural_group2 = 1;
if(strncmp(IP_line2.C3_element," CR ",4)==0) structural_group2 = 2;

//SECONDARY BIN ASSIGNMENT:
int SEC_residueBIN, SEC_structuralBIN;

for(i=0,RES_COMBOS=0;i<RES_GROUPS;++i)  {
  for(j=i;j<RES_GROUPS;++j)  {
    if(i==residue_group1 && j==residue_group2) SEC_residueBIN=RES_COMBOS;
    if(i==residue_group2 && j==residue_group1) SEC_residueBIN=RES_COMBOS;
    ++RES_COMBOS;
    }
  }

for(i=0,STR_COMBOS=0;i<STR_GROUPS;++i)  {
  for(j=i;j<STR_GROUPS;++j)  {
    if(i==structural_group1 && j==structural_group2) SEC_structuralBIN=STR_COMBOS;
    if(i==structural_group2 && j==structural_group1) SEC_structuralBIN=STR_COMBOS;
    ++STR_COMBOS;
    }
  }

//FINAL BIN ASSIGNMENT:
int final_bin=0;
for(i=0;i<RES_COMBOS;++i)  {
  for(j=0;j<STR_COMBOS;++j)  {
    if(SEC_residueBIN==i && SEC_structuralBIN==j) return final_bin;
    ++final_bin;
    }
  }

//Return FAIL-VALUE, If no binning was carried out within this function:
return -1;
}






