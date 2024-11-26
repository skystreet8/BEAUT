/*Step0-END-FILE.c:
This program prints a unique [END-FILE][INT_MAX][INT_MAX] file-delimiter that is 
concatenated to the end of concatenated cabbage-files
*/
#include<stdio.h>
#include<limits.h>

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

//MAIN FUNCTION:
int main()
{
int i;
UNION_int temp_intUNION;
temp_intUNION.i = INT_MAX;
printf("END-FILE");

for(i=0;i<sizeof(int);++i)  {
  printf("%c", temp_intUNION.c[i]);
  }
//Debugging: produce wrong END-OF-FILE integer:
//temp_intUNION.i = INT_MAX-1;
for(i=0;i<sizeof(int);++i)  {
  printf("%c", temp_intUNION.c[i]);
  }

return 0;
}
