/*Step0-cabbage_decoder.c:
This program translates binary cabbage-files into human-readable strings.
The output of this program CANNOT be used for further calculations.
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
#include"Step0-PDBclass.h"
#include"Step0-cabbage_core.h"

//MAIN FUNCTION:
int main(int argc, char *argv[])
{
//Standard error message:
if(argc!=2)  {
  printf("usage: ./Step0-cabbage_decoder <cabbage unit file>\n");
  return 0;
  }

FILE *cabbageIP=fopen(argv[1], "r");
int i, j, k, bin_counter[90];
char c;
element elem;
UNION_int temp_intUNION;
UNION_double temp_doubleUNION;

//Print header-data:
printf("Step0-cabbage_decoder.cpp (executable): Part of \"PM_pipeline\"\n");

//Print file-length:
for(i=0;i<sizeof(int);++i)  {
  temp_intUNION.c[i] = fgetc(cabbageIP);
  }
printf("This file is %8d bytes long\n", temp_intUNION.i);

//Print file-name:
printf("INPUT file-name: \"");
for(i=0;i<OP_NAME_SIZE;++i) printf("%c", fgetc(cabbageIP));
printf("\"\n");

//Print number of <point-type-comparisons> per bin:
printf("6 x 15 <point-type-comparisons>:\n\n");
for(i=0;i<90;++i)  {
  for(j=0;j<sizeof(int);++j) temp_intUNION.c[j]=fgetc(cabbageIP);
  printf("%8d ", temp_intUNION.i);
  bin_counter[i] = temp_intUNION.i;
  if((i+1)%6==0) printf("\n");
  }
printf("\n");

//Print individual <distance elements>, residue-origins:
for(i=0;i<90;++i)  {
  printf("<distance-elements> in bin: %2d\n", i);
  for(j=0;j<bin_counter[i];++j)  {
    //Read all data for structure 'element':
    for(k=0;k<sizeof(double);++k) elem.distance.c[k]=fgetc(cabbageIP);
    for(k=0;k<sizeof(int);++k) elem.residue1.c[k]=fgetc(cabbageIP);
    for(k=0;k<sizeof(int);++k) elem.residue2.c[k]=fgetc(cabbageIP);
    //Print all data from structure 'element':
    printf("%8.3f %8d %8d\n", elem.distance.d, elem.residue1.i, elem.residue2.i);
    }
  printf("\n");
  }
printf("END-OF-FILE\n");
return 0;
}


















