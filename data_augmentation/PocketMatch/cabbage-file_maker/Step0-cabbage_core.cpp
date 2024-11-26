/*Step0-cabbage-core.cpp:
This step converts .pdb pocket files into <cabbage unit files> (binary)
These binary files can then be fed into any version of PocketMatch for further calculation
*/

//C++ standard libraries:
#include<iostream>
#include<cstdlib>
#include<memory>
#include<new>
#include<vector>
#include<algorithm>

//C header files:
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<limits.h>

//Custom definitions:
#define CHAIN_FACTOR 100000
#define OP_NAME_SIZE 32
#define DEBUG 0

//Custom header files:
//NOTE: "Step0-cabbage.h" is dependent on "Step0-PDBclass.h". Maintain this order.
#include"Step0-PDBclass.h"
#include"Step0-cabbage_core.h"



using namespace std;

//MAIN FUNCTION:
int main(int argc, char *argv[])
{
//Standard error message:
if(argc!=2)  {
  printf("usage: ./Step0-cabbage <input-pocket.pdb>\n");
  return 0;
  }

//Initialise/declare program variables:
FILE *IP_file=fopen(argv[1], "r"), *OP_file;
int i, j, k, no_lines=0, no_res=0;
char IP_line[81];

//Determine how many "ATOM  " lines the input-file contains":
while(1)  {
  for(i=0;i<81;++i)  {
    IP_line[i] = fgetc(IP_file);
    if(IP_line[i]=='\n') break;
    if(IP_line[i]==EOF) break;
    }
  if(IP_line[i]==EOF) break;
  if(strncmp("ATOM  ", IP_line, 6)==0) ++no_lines;
  }
fseek(IP_file, 0, SEEK_SET);

//Allocate memory for "ATOM  " lines:
PDBclass *PDBtable=new PDBclass[no_lines];
//Fill PDBtable:
for(i=0;i<no_lines;)  {
  for(j=0;j<81;++j)  {
    IP_line[j] = fgetc(IP_file);
    if(IP_line[j]=='\n') break;
    if(IP_line[j]==EOF) break;
    }
  if(IP_line[j]==EOF) break;
  if(strncmp(IP_line, "ATOM  ", 6)==0)  {
    PDBtable[i].get_line(IP_line);
    //Count number of amino-acid residues present in input: 
    //if(strncmp(PDBtable[i].C3_element, " CA ",4)==0) ++no_res;
    PDBtable[i].res_name_shorten();
    ++i;
    }
  }

//Remove 'altloc' atoms:
no_altloc(PDBtable, &no_lines);


//Calculate number of residues present in input:
for(i=0;i<no_lines;++i)  {
  if(strncmp(PDBtable[i].C3_element, " CA ",4)==0) ++no_res;
  }

//Allocate memory for CaCbCR lines:
PDBclass *PDB_CaCbCR=new PDBclass[no_res*3];
//Calculate CA, CB, centroids. write into new PDBclass:
int OLD_resno=0, NEW_resno=0;
for(i=0,j=0;i<no_lines;++i)  {
  NEW_resno = PDBtable[i].C6_residue;
  if(NEW_resno!=OLD_resno)  {
    CaCbCR(PDBtable+i, PDB_CaCbCR+(j*3));
    OLD_resno = NEW_resno;
    ++j;
    }
  }

//Initialise array to count number of <pairwise-distances> per bin:
int bin_counter[90];
bzero(bin_counter, 90*sizeof(int));

//Determine memory to be allocated per bin:
double distance;
int bin;
for(i=0;i<no_res*3;++i)  {
  for(j=(i+1);j<no_res*3;++j)  {
    //Determine if CB is present in the input lines;
    if(strncmp(PDB_CaCbCR[i].C1_name,"PSEUDO",6)!=0 && strncmp(PDB_CaCbCR[j].C1_name,"PSEUDO",6)!=0)  {
      //Assign bin to <pairwise-distance>, count in 'bin_counter':
      bin = binner(PDB_CaCbCR[i], PDB_CaCbCR[j]);
      if(bin!=-1) ++bin_counter[bin];
      }
    }
  }

/*Allocate memory required per bin, based on values in 'bin_counter[90]': Memory allocation is 
done using a double-pointer **bin_vals. It points to a 90-unit-long malloc'ed-array containing
pointers to more malloc'ed arrays that store binned {<pairwise distances>, 'line1->C6_residue' and
'line2->C6_residue' values}, as the 'element' datatype. 

I admit that this data-structure is unnessessarily complex. A better description of my data-structure
can be found in the given documentation.
I could have just used a malloc'ed array with starting positions stored in a separate pointer array.
*/
element **bin_vals=(element**)malloc(90*sizeof(element*));
for(i=0;i<90;++i)  {
  *(bin_vals+i) = (element*)malloc(bin_counter[i]*sizeof(element));
  }

//Initialise 'bin_counter2'. 'bin_counter' and 'bin_counter2' should ultimately have identical values:
int bin_counter2[90];
bzero(bin_counter2, 90*sizeof(int));

//Calculate <pairwise-distances>, assign values to bins:
for(i=0;i<no_res*3;++i)  {
  for(j=(i+1);j<no_res*3;++j)  {
    //Determine if CB is present in the input lines;
    if(strncmp(PDB_CaCbCR[i].C1_name,"PSEUDO",6)!=0 && strncmp(PDB_CaCbCR[j].C1_name,"PSEUDO",6)!=0)  {
      //Assign bin to <pairwise-distance>:
      bin = binner(PDB_CaCbCR[i], PDB_CaCbCR[j]);
      if(bin!=-1)  {
        //transfer <pairwise-distance>, residue numbers to 'element':
        *(*(bin_vals+bin)+bin_counter2[bin]) = make_element(PDB_CaCbCR[i], PDB_CaCbCR[j]);
        ++bin_counter2[bin];
        }
      }
    }
  }

//Error message:
for(i=0;i<90;++i)  {
  if(bin_counter[i]!=bin_counter2[i])  {
    printf("<Step0-cabbage> Error: bin_counter[%d]!=bin_counter2[%d]\n", i, i);
    printf("bin read/write error caused program termination\n");
    return 0;
    }
  }

//Sort binned <pairwise-distances> in 'element' datatype:
//I'm using a custom-implementation of bubble-sort:
bool bubble_end;
element temp_elem;

for(i=0;i<90;++i)  {
  while(1)  {
    bubble_end = 0;
    for(j=0;j<bin_counter[i]-1;++j)  {
      if( (*(bin_vals+i)+j)->distance.d > (*(bin_vals+i)+j+1)->distance.d )  {
        //Switch positions of the 2 'elements':
        temp_elem = *(*(bin_vals+i)+j);
        *(*(bin_vals+i)+j) = *(*(bin_vals+i)+j+1);
        *(*(bin_vals+i)+j+1) = temp_elem;
        bubble_end = 1;
        }
      }
    if(bubble_end==0) break;
    }
  }

bool output_format=1;
/*Conventional Print output: Compatible with Kali's/my-old PocketMatch implementation.
Flip the 'output_format' variable to '0' to print "Conventional Print output".
Flip the 0/1 'DEBUG' definition for easy debugging.
*/
if(output_format==0 && DEBUG==0)  {
  printf("15\n"); //total number of combinations of residue groups
  printf("6\n");  //total number of combinations of structural groups
  for(i=0;i<90;++i)  {
    printf("%d\n", bin_counter[i]);
    for(j=0;j<bin_counter[i];++j)  {
      printf("%.6f ", (*(bin_vals+i)+j)->distance.d );
      }
    printf("\n");
    }
  }

/*New Print output: Incompatible with Kali's/my-old PocketMatch implementation.
The new format stores residue-origins of <pairwise-distances> for further calculations.
Flip the 'output_format' variable to '0' to print "Conventional Print output".
Flip the 0/1 'DEBUG' definition for easy debugging.
*/
UNION_int temp_intUNION, file_intUNION;
int BOF_length=0, name_length=0, bin_length=0, element_length=0, EOF_length=0;
char OP_name[OP_NAME_SIZE], *argv_start, *argv_end;

if(output_format==1 && DEBUG==0)  {
  //Determine, print file-length (as number of bytes):
  BOF_length = sizeof(int);
  name_length = OP_NAME_SIZE*sizeof(char);
  bin_length = 90*sizeof(int);
  for(i=0;i<90;++i)  {
    element_length += bin_counter[i]*sizeof(element);
    }
  EOF_length = sizeof(int);
  file_intUNION.i = BOF_length +name_length +bin_length +element_length +EOF_length;
  for(i=0;i<sizeof(int);++i) printf("%c", file_intUNION.c[i]);

  //Determine input-file name, print along with output:
  //Initialise values to OP_name array to '_':
  for(i=0;i<OP_NAME_SIZE;++i) OP_name[i]='_';
  //Determine start/end of argv[1] (input-file-name):
  for(i=0;;++i)  {
    if(*(argv[1]+i)=='/')  {
      argv_start = argv[1]+i+1; //START printing AFTER '/' character
      }
    if(*(argv[1]+i)==0)  {
      argv_end = argv[1]+i-1; //STOP printing BEFORE null character
      break;
      }
    }
  //Feed 'argv_start/end' character-string into 'OP_name':
  for(i=0;i<OP_NAME_SIZE;++i)  {
    OP_name[i] = *(argv_start+i);
    if(argv_start+i==argv_end) break;
    }
  //Print OP_name' on terminal:
  for(i=0;i<OP_NAME_SIZE;++i) printf("%c", OP_name[i]);

  //Print memory usage of bins:
  for(i=0;i<90;++i)  {
    temp_intUNION.i = bin_counter[i];
    for(j=0;j<sizeof(int);++j) printf("%c", temp_intUNION.c[j]);
    }

  for(i=0;i<90;++i)  {
    for(j=0;j<bin_counter[i];++j)  {
      //Print <pairwise-distance>:
      for(k=0;k<sizeof(double);++k) printf("%c", (*(bin_vals+i)+j)->distance.c[k] );
      //Print residue-origins of <pairwise-distance>:
      for(k=0;k<sizeof(int);++k) printf("%c", (*(bin_vals+i)+j)->residue1.c[k] );
      for(k=0;k<sizeof(int);++k) printf("%c", (*(bin_vals+i)+j)->residue2.c[k] );
      }
    }
  }
//Print end-of-file integer:
temp_intUNION.i = INT_MIN;
for(i=0;i<sizeof(int);++i) printf("%c", temp_intUNION.c[i]);

return 0;
}







