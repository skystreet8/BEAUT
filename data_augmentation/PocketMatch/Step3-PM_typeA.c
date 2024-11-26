/* Step3-PMserial.c:
This step carries out the actual PocketMatch computation.
All <cabbage unit files> are concatenated, compressed, and fed as <cabbage datafile: binary>
The program outputs to the terminal, and can be piped/saved for further use
SERIAL, ALL-TO-ALL comparisons are c/o. Use other versions of this program for other comparisons
*/
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<limits.h>

//DEFINE cutoff: The cutoff for pairwise length comparisons is defined as 5Å.
//Change this default and recompile to use a different cutoff:
#define CUTOFF 0.5

//DEFINE length of output PM-names: PM_pipeline reads <cabbage unit file> names.
//Names are outputted alongside pairwise PM scores:
#define OP_NAME_SIZE 32

//DEFINE whether SUMA_score is calculated: (1/0) (yes/no):
int SUMA_TOGGLE=0;

//GLOBAL output line-length
//(if SUMA_TOGGLE==0) 111:
//(if SUMA_TOGGLE==1) 120:
int LINE_LENGTH=0;

//DEFINE sizeof SUMA-block:
#define BLOCK_SIZE 704

//DEFINE denominator used for residue-chain character-extraction:
#define CHAIN_FACTOR 100000

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

//STRUCTURE: 'twin_int' is used to store residue-numbers of <pairwise distance> matches: 
typedef struct  {
  int i[2];
  }twin_int;

//Custom header files:
#include "Step3-PM.h"

//FUNCTION: Check if 'END-OF-FILE' is reached:
int check_end(FILE *fileIP)
{
//Reset file-pointer back to 'E' in 'END-FILE':
fseek(fileIP, -1, SEEK_CUR);

UNION_int temp_intUNION;
int i;
char end_file[8];

//Check first 2 words (64 bits in x86) for EOF statement:
for(i=0;i<8;++i) end_file[i]=fgetc(fileIP);
if(strncmp(end_file, "END-FILE", 8)!=0)  {
  //Return file-pointer:
  fseek(fileIP, -7, SEEK_CUR);
  return 0;
  }
//Check third word for INT_MAX;
for(i=0;i<sizeof(int);++i) temp_intUNION.c[i]=fgetc(fileIP);
if(temp_intUNION.i!=INT_MAX)  {
  //Return file-pointer:
  fseek(fileIP, -7-sizeof(int), SEEK_CUR);
  return 0;
  }
//Check fourth word for INT_MAX;
for(i=0;i<sizeof(int);++i) temp_intUNION.c[i]=fgetc(fileIP);
if(temp_intUNION.i!=INT_MAX)  {
  //Return file-pointer:
  fseek(fileIP, -7-2*sizeof(int), SEEK_CUR);
  return 0;
  }
//Return file-pointer to original location:
fseek(fileIP, -7-2*sizeof(int), SEEK_CUR);
return 1;
}

//FUNCTION: The 'match' function carries out pairwise PocketMatch scoring.
//It reads 2 <cabbage unit files> and directly prints a character string of results.
int match(char *file1_const, char *file2_const,
          FILE *OP_file1, FILE *OP_file2, 
          int FILE_address1, int FILE_address2)
{
/* Variable explanations:
  *file1: pointer for <cabbage unit file-1>. NOT a file-pointer
  *file1-const: input pointer, used to to keep track of the beginning of *file1
  *file2: pointer for <cabbage unit file-2>. NOT a file-pointer
  *file2-const: input pointer, used to to keep track of the beginning of *file2
  i, j: loop variables
  SUM_file1/valsum1: Total number of <distance-elements> in <cabbage unit file-1>
  SUM_file2/valsum2: Total number of <distance-elements> in <cabbage unit file-2>
  counter: counts the number of matching <distance-elements> between <cabbage unit files-1/2>

  frac1: P-min (PocketMatch minimum) score
  frac2: P-max (PocketMatch maximum) score
  bin_counter1: array[1x90] of no. of <pairwise distances> per <point-type-comparison> in <cabbage unit file-1>
  bin_counter2: array[1x90] of no. of <pairwise distances> per <point-type-comparison> in <cabbage unit file-2>
  num1end: Number of <pairwise distances> in a given <point-type-comparison> in <cabbage unit file-1>
           Extracted from val1 array[1x90]
  num2end: Number of <pairwise distances> in a given <point-type-comparison> in <cabbage unit file-2>
           Extracted from val1 array[1x90]
  num1: Incremental counter, counts from 0 to num1end
  num2: Incremental counter, counts from 0 to num2end
  name1: User-given file-name of <cabbage unit file-1>
  name2: User-given file-name of <cabbage unit file-2>
*/

int i, j;
int bin_counter1[90], bin_counter2[90];
char *file1=file1_const, *file2=file2_const;
char name1[OP_NAME_SIZE], name2[OP_NAME_SIZE];
UNION_int temp_intUNION;

//Read filename of <cabbage unit files-1/2>
for(i=0;i<OP_NAME_SIZE;++i)  {
  name1[i] = *(file1 +i +sizeof(int));
  name2[i] = *(file2 +i +sizeof(int));
  }

//Shift 'file-pointers' to appropriate location:
file1 = file1 +OP_NAME_SIZE +sizeof(int);
file2 = file2 +OP_NAME_SIZE +sizeof(int);

//Read number of <distance-elements>:
//Calculate total number of <distance-elements> for 'file1/2':
int SUM_file1=0, SUM_file2=0;
for(i=0;i<90;++i)  {
  bin_counter1[i] = *((int*)file1+i);
  SUM_file1 += bin_counter1[i];
  bin_counter2[i] = *((int*)file2+i);
  SUM_file2 += bin_counter2[i];
  }
//Shift 'file-pointers' to appropriate location:
file1 = file1+sizeof(int)*90;
file2 = file2+sizeof(int)*90;
//Declare 'element' file-pointers at start of <distance-elements> block:
element *file1_element=(element*)file1, *file1_element_const=(element*)file1;
element *file2_element=(element*)file2, *file2_element_const=(element*)file2;

//Initialise hash-table to store residues of <distance-elements> that pass CUTOFF threshold:
twin_int *hash_table1, *hash_table2;
int hash_size;
if(SUMA_TOGGLE==1)  {
  if(SUM_file1>SUM_file2) hash_size=SUM_file1;
  else hash_size=SUM_file2;
  hash_size = ((int)((sqrt(8*hash_size+1)+1)/2))+1;
  //Since 2-residue numbers accompany 1 <distance-element> the hash-table
  //size has to be a function of the largest number of <distance-elements>
  hash_table1 = make_hash(hash_size);
  hash_table2 = make_hash(hash_size);
  }

/*===================================\
|      PocketMatch Calculation       |
\===================================*/
int valsum1=0, valsum2=0, counter=0, num1end, num2end;
double frac1, frac2;
element num1, num2;
int IP_value1[3], IP_value2[3];

for(i=0;i<90;++i)  {
  valsum1 = valsum1+bin_counter1[i];
  valsum2 = valsum2+bin_counter2[i];
  if((bin_counter1[i]!=0)&&(bin_counter2[i]!=0))  {
    num1end = bin_counter1[i];
    num2end = bin_counter2[i];
    //Read <distance-elements> values, Increment file-pointers:
    num1 = *file1_element; ++file1_element;
    num2 = *file2_element; ++file2_element;

    while(1)  {
      //Count number of <distance-elements> that pass CUTOFF threshold:
      if(num1.distance.d-num2.distance.d<0.5 && num1.distance.d-num2.distance.d>-0.5)  {
        if(SUMA_TOGGLE==1)  {
          //Store 'residue-numbers' in 'hash_table1':
          IP_value1[0] = num1.residue1.i;
          IP_value1[1] = num2.residue1.i;
          IP_value1[2] = num2.residue2.i;
          IP_hash(hash_table1, IP_value1);

          IP_value1[0] = num1.residue2.i;
          IP_hash(hash_table1, IP_value1);

          //Store 'residue-numbers' in 'hash_table2':
          IP_value2[0] = num2.residue1.i;
          IP_value2[1] = num1.residue1.i;
          IP_value2[2] = num1.residue2.i;
          IP_hash(hash_table2, IP_value2);

          IP_value2[0] = num2.residue2.i;
          IP_hash(hash_table2, IP_value2);
          }
        ++counter;
        --num1end;
        --num2end;

        //Read <distance-elementse> values, Increment file-pointers:
        num1 = *file1_element; ++file1_element;
        num2 = *file2_element; ++file2_element;
        }
      else  {
        if(num1.distance.d<num2.distance.d)  {
          --num1end;
          //Read <distance-elements> value, Increment file-pointer:
          num1 = *file1_element; ++file1_element;
          }
        else  {
          --num2end;
          //Read <distance-elements> value, Increment file-pointer:
          num2 = *file2_element; ++file2_element;
          }
        }
      if(num1end<=0) break;
      if(num2end<=0) break;
      }
    }
  //Move onto another set of <point-type-comparisons>
  file1_element = file1_element_const+valsum1;
  file2_element = file2_element_const+valsum2;
  }

//Assign values to frac1 (P-min) and frac2 (P-max) based on 
//total number of <distance-elements> in <cabbage unit files-1/2>
if(valsum1<valsum2)  {
  frac1 = (double)counter/(double)valsum1;
  frac2 = (double)counter/(double)valsum2;
  }
else  {
  frac1 = (double)counter/(double)valsum2;
  frac2 = (double)counter/(double)valsum1;
  }

/*===================================\
|      SUMA-SCORE: Calculations      |
\===================================*/
int SUMA_score=0;
char SUMA_block[BLOCK_SIZE];

if(SUMA_TOGGLE==1)  {
  strncpy(SUMA_block, name1, 32);
  strncpy(SUMA_block+32, name2, 32);
  compare_hash(hash_table1, hash_table2, &SUMA_score, SUMA_block);
  }

/*===================================\
|           Output results           |
\===================================*/
//Print output string (1 line only):
//If one or more input pockets has no matches:
char *OP_string=malloc(LINE_LENGTH*sizeof(char));

if(valsum1==0||valsum2==0)  {
  if(SUMA_TOGGLE==1)  {
    sprintf(OP_string, "%.32s %.32s NULL____ NULL____ ________ ________ ________ ________\n", name1, name2);
    }
  else  {
    sprintf(OP_string, "%.32s %.32s NULL____ NULL____ ________ ________ ________\n", name1, name2);
    }
  }
//Standard output:
else  {
  if(SUMA_TOGGLE==1)  {
  sprintf(OP_string, "%.32s %.32s %8.6f %8.6f %8d %8d %8d %8d\n", name1, name2, frac1, frac2, valsum1, valsum2, counter, SUMA_score);
    }
  else  {
  sprintf(OP_string, "%.32s %.32s %8.6f %8.6f %8d %8d %8d\n", name1, name2, frac1, frac2, valsum1, valsum2, counter); 
    }
  }

//Write outputs to PM-score file, 'OP_file1':
fseek(OP_file1, FILE_address1, SEEK_SET);
for(i=0;i<LINE_LENGTH;++i) fputc(*(OP_string+i), OP_file1);

//Write outputs to SUMA-score file, 'OP_file2':
fseek(OP_file2, FILE_address2, SEEK_SET);
for(i=0;i<BLOCK_SIZE;++i) fputc(SUMA_block[i], OP_file2);

if(SUMA_TOGGLE==1)  {
  //Free the 2 hash-tables' allocated memory:
  free(hash_table1);
  free(hash_table2);
  }

//Free all allocated memory:
free(OP_string);
return 0;
}

/*===================================\
|            MAIN FUNCTION           |
\===================================*/
int main(int argc, char *argv[])
{
//Standard error message:
if(argc!=2)  {
  printf("usage: ./Step3-PMserial <outfile.cabbage>>\n");
  return 0;
  }

//make sure that the user does NOT use the sphinx module:
SUMA_TOGGLE=0;
LINE_LENGTH=111+SUMA_TOGGLE*9;

/* Variable explanations:
  *fileIP: input pointer for <cabbage datafile: binary>
  i,j: loop variables
  LENGTH_file: length of <cabbage datafile: binary>
  NUMBER_files: number of <cabbage unit files> in <cabbage datafile: binary>
  *Mfile: *fileIP is read to main-memory to hasten data retrieval
  **address: A list of locations on *Mfileconst corresponding to the beginning of individual cabbage files
*/
int i, j, k, LENGTH_file=0, NUMBER_files=0;
UNION_int subLENGTH_file;

//Declare *fileIP, read <cabbage datafile: binary> into main memory:
FILE *fileIP;
fileIP = fopen(argv[1], "r");

//Find total length of <cabbage datafile: binary> (LENGTH_file):
//Also find number of <cabbage unit files> (NUMBER_files):
while(1)  {
  //Read file-size of <cabbage unit file>:
  for(i=0;i<sizeof(int);++i)  {
    subLENGTH_file.c[i] = fgetc(fileIP);
    if(subLENGTH_file.c[i]=='E')  {
      if(check_end(fileIP)==1)  {
        break;
        }
      }
    }
  if(check_end(fileIP)==1) break;
  LENGTH_file += subLENGTH_file.i;
  ++NUMBER_files;
  //Move to new file-size integer:
  fseek(fileIP, subLENGTH_file.i-sizeof(int), SEEK_CUR);
  }
fseek(fileIP, 0, SEEK_SET);

//Allocate memory to *Mfile to store entire <cabbage datafile: binary>:
char *Mfile;
Mfile = (char*)malloc(LENGTH_file*sizeof(char));

//Read entire <cabbage datafile: binary> into memory:
for(i=0;i<LENGTH_file;++i)  {
 *(Mfile+i) = fgetc(fileIP); 
  }
fseek(fileIP, 0, SEEK_SET);

//Allocate memory to **addressconst to store addresses of all the <cabbage unit files>:
void **address;
address = (void**)malloc(sizeof(void*)*NUMBER_files);
for(i=0;i<NUMBER_files;++i)  {
  *(address+i) = 0;
  }

//Read all <cabbage unit file> start-pointers into malloc'ed **address:
int shift=0;
for(i=0;i<NUMBER_files;++i)  {
  //Write start of <cabbage unit file> into **address pointer:
  *(address+i) = Mfile+shift;
  //Find byte-shift from current-file to next-file:
  shift += *((int*)Mfile+shift/sizeof(int));
  }

//Cleanup: Return allocated file-pointer:
fclose(fileIP);

//RDS: Select all combinations of <cabbage unit files>, run PocketMatch:
FILE *OP_file1=fopen("PocketMatch_score.txt", "w+");
FILE *OP_file2=fopen("PocketMatch_pairs.txt", "w+");
int FILE_address1, FILE_address2;

for(i=0;i<NUMBER_files;++i)  {
  for(j=i+1;j<NUMBER_files;++j)  {
    //Call 'match' function for selected <cabbage unit file> pairs:
    FILE_address1 = ((NUMBER_files*i) -(i*(i+1))/2 +(j-i-1))*LINE_LENGTH;
    FILE_address2 = ((NUMBER_files*i) -(i*(i+1))/2 +(j-i-1))*BLOCK_SIZE;
    match(*(address+i), *(address+j), OP_file1, OP_file2, FILE_address1, FILE_address2);
    }
  }

//delete the "PocketMatch_pairs.txt file. It's not needed in this version:
remove("PocketMatch_pairs.txt");

//END program:
return 0;
}


