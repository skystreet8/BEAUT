/*Step-0-PDBclass.h (header file):
This file contains the object 'PDBclass' that allows easy IP/OP & calculations on PDB lines.
'PDBclass' is generic. I encourage interested bioinformatics programmers (if any) to adopt 
and build upon this object.
*/

using namespace std;

//OBJECT: PDBclass to facilitate easy IP/OP:
class PDBclass
{
public:
//numbers:
  int  C2_atom;
  int  C6_residue;
  double C7_coords[3];
  double C10_occupy;
  double C11_temp;
//characters:
  char C1_name[6];
  char C12_element[2];
  char C3_element[4];
  char C4_aminoacid[3];
  char C5_chain;
  char C4_short;
  char C_altloc;
//functions:
  void get_line(char*);
  void put_line();
  void res_name_shorten();
  void clean();
  friend double dist(PDBclass, PDBclass);
  friend void CaCbCR(PDBclass*, PDBclass*);
  friend void no_altloc(PDBclass*, int*);
};

//FUCTION: read PDB-file from file-pointer into 'object' PDBclass;
void PDBclass::get_line(char *line)
{
//read characters:
char number[8];
int i;
strncpy(C1_name, line, 6);
strncpy(C3_element, line+12, 4);
C_altloc = *(line+16);
strncpy(C4_aminoacid, line+17, 3);
C5_chain = *(line+21);
strncpy(C12_element, line+76,2);

//read numbers:
strncpy(number, "        ", 8);
strncpy(number, line+6, 5);
C2_atom = atoi(number);

strncpy(number, "        ", 8);
strncpy(number, line+22, 4);
C6_residue = atoi(number);

strncpy(number, "        ", 8);
strncpy(number, line+30, 8);
C7_coords[0] = atof(number);
strncpy(number, "        ", 8);
strncpy(number, line+38, 8);
C7_coords[1] = atof(number);
strncpy(number, "        ", 8);
strncpy(number, line+46, 8);
C7_coords[2] = atof(number);

strncpy(number, "        ", 8);
strncpy(number, line+54, 6);
C10_occupy = atof(number);

strncpy(number, "        ", 8);
strncpy(number, line+60, 6);
C11_temp = atof(number);
}

//FUCTION: print PDB-line;
void PDBclass::put_line()
{
int i;
for(i=0;i<6;++i) printf("%c", C1_name[i]);
printf("%5d ", C2_atom);
for(i=0;i<4;++i) printf("%c", C3_element[i]);
printf("%c", C_altloc);
for(i=0;i<3;++i) printf("%c", C4_aminoacid[i]);
printf(" %c%4d    ", C5_chain, C6_residue);
printf("%8.3f%8.3f%8.3f", C7_coords[0], C7_coords[1], C7_coords[2]);
printf("%6.2f%6.2f          ", C10_occupy, C11_temp);
printf("%c%c  \n", C12_element[0], C12_element[1]);
}

//FUNCTION: Convert amino-acid names:
void PDBclass::res_name_shorten()
{
if(strncmp(C4_aminoacid, "ALA", 3)==0) C4_short='A';
if(strncmp(C4_aminoacid, "ARG", 3)==0) C4_short='R';
if(strncmp(C4_aminoacid, "ASN", 3)==0) C4_short='N';
if(strncmp(C4_aminoacid, "ASP", 3)==0) C4_short='D';
if(strncmp(C4_aminoacid, "CYS", 3)==0) C4_short='C';
if(strncmp(C4_aminoacid, "GLN", 3)==0) C4_short='Q';
if(strncmp(C4_aminoacid, "GLU", 3)==0) C4_short='E';
if(strncmp(C4_aminoacid, "GLY", 3)==0) C4_short='G';
if(strncmp(C4_aminoacid, "HIS", 3)==0) C4_short='H';
if(strncmp(C4_aminoacid, "ILE", 3)==0) C4_short='I';
if(strncmp(C4_aminoacid, "LEU", 3)==0) C4_short='L';
if(strncmp(C4_aminoacid, "LYS", 3)==0) C4_short='K';
if(strncmp(C4_aminoacid, "MET", 3)==0) C4_short='M';
if(strncmp(C4_aminoacid, "PHE", 3)==0) C4_short='F';
if(strncmp(C4_aminoacid, "PRO", 3)==0) C4_short='P';
if(strncmp(C4_aminoacid, "SER", 3)==0) C4_short='S';
if(strncmp(C4_aminoacid, "THR", 3)==0) C4_short='T';
if(strncmp(C4_aminoacid, "TRP", 3)==0) C4_short='W';
if(strncmp(C4_aminoacid, "TYR", 3)==0) C4_short='Y';
if(strncmp(C4_aminoacid, "VAL", 3)==0) C4_short='V';
}

//FUNCTION: Clean line:
void PDBclass::clean()
{
//Clean characters:
strncpy(C1_name, "      ", 6);
strncpy(C3_element, "      ", 4);
C_altloc = ' ';
strncpy(C4_aminoacid, "   ", 3);
C5_chain = ' ';
strncpy(C12_element, "  ",2);

//Clean numbers:
C2_atom = 0;
C6_residue = 0;
C7_coords[0] = 0;
C7_coords[1] = 0;
C7_coords[2] = 0;
C10_occupy = 0;
C11_temp = 0;
}

//FUNCTION: Calculate distance between 2 PDB lines:
double dist(PDBclass IP_line1, PDBclass IP_line2)
{
double distance;
distance = sqrt( pow(IP_line1.C7_coords[0]-IP_line2.C7_coords[0],2)
                +pow(IP_line1.C7_coords[1]-IP_line2.C7_coords[1],2)
                +pow(IP_line1.C7_coords[2]-IP_line2.C7_coords[2],2)
               );
return distance;
}

//FUNCTION: Calculate CA, CB, centroids. write into output-PDBclass:
void CaCbCR(PDBclass *IP_table, PDBclass *OP_table)
{
int res_no=0, i, j, k, l, res_length=0, CR_flag=0, ALA_flag=0;
double CA_counter=0, CB_counter=0, CR_counter=0;

//Read IP_table-pointer, determine lines occupied by given residue:
for(i=0;;++i)  { 
  res_no = IP_table[i].C6_residue;
  if(res_no==IP_table[1].C6_residue) ++res_length;
  else break;
  }

//Clean output PDBclass:
OP_table[0].clean();
OP_table[1].clean();
OP_table[2].clean();

//Copy CA, CB, to OP_table:
//Loops & averaging account for multiple CA, CB, entries in case of 'altloc' indicators:

//LINE-1: Read first CA-atom:
for(i=0;i<res_length;++i)  {
  if(strncmp(IP_table[i].C3_element, " CA ", 4)==0)  {
    OP_table[0] = IP_table[i];
    ++CA_counter;
    break;
    }
  }
  //Check for absence of CA-atom;
  if(CA_counter==0)  {
    strncpy(OP_table[0].C1_name,"PSEUDO",6);
    strncpy(OP_table[0].C3_element, " CA ", 4);
    strncpy(OP_table[0].C4_aminoacid, IP_table[0].C4_aminoacid, 3);
    OP_table[0].C4_short = IP_table[0].C4_short;
    strncpy(OP_table[0].C12_element, " X", 2);
  }

//LINE-2: Read first CB-atom:
for(i=0;i<res_length;++i)  {
  if(strncmp(IP_table[i].C3_element, " CB ", 4)==0)  {
    OP_table[1] = IP_table[i];
    ++CB_counter;
    break;
    }
  }
  //Check for absence of CA-atom;
  if(CB_counter==0)  {
    strncpy(OP_table[1].C1_name,"PSEUDO",6);
    strncpy(OP_table[1].C3_element, " CB ", 4);
    strncpy(OP_table[1].C4_aminoacid, IP_table[0].C4_aminoacid, 3);
    OP_table[1].C4_short = IP_table[0].C4_short;
    strncpy(OP_table[1].C12_element, " X", 2);
  }

//Check all input lines for duplicate atoms (alternate location indicator):
char altloc_flag=' ';
for(i=0;i<res_length;++i)  {
  //Pick up first altloc atom-set, break loop:
  if(IP_table[i].C_altloc!=altloc_flag)  {
    altloc_flag = IP_table[i].C_altloc;
    break;
    }
  }

//Check if residue is ALANINE:
if(IP_table[1].C4_short=='A') ALA_flag=1;

/*OPTIONAL: If CB absent (gly), compensate by vectorial extrapolation:
I have not extrapolated here as Kalidas didn't.
*/

if(ALA_flag==1)  {
  //Mark the "CR" point as absent:
  strncpy(OP_table[2].C1_name, "PSEUDO", 6);
  strncpy(OP_table[2].C3_element, " CR ", 4);
  strncpy(OP_table[2].C4_aminoacid, IP_table[0].C4_aminoacid, 3);
  strncpy(OP_table[2].C12_element, " X", 2);
  }
else  {
  //Calculate centroid co-ordinates:
  for(i=0;i<res_length;++i)  {
    CR_flag = 0;
    //Check atom-name, altloc-flag:
    if(strncmp(IP_table[i].C3_element, " CA ", 4)!=0 && IP_table[i].C_altloc==altloc_flag) ++CR_flag;
    if(strncmp(IP_table[i].C3_element, " CB ", 4)!=0 && IP_table[i].C_altloc==altloc_flag) ++CR_flag;
    if(strncmp(IP_table[i].C3_element, " N  ", 4)!=0 && IP_table[i].C_altloc==altloc_flag) ++CR_flag;
    if(strncmp(IP_table[i].C3_element, " C  ", 4)!=0 && IP_table[i].C_altloc==altloc_flag) ++CR_flag;
    if(strncmp(IP_table[i].C3_element, " O  ", 4)!=0 && IP_table[i].C_altloc==altloc_flag) ++CR_flag;
    if(CR_flag==5)  {
      OP_table[2].C7_coords[0] += IP_table[i].C7_coords[0];
      OP_table[2].C7_coords[1] += IP_table[i].C7_coords[1];
      OP_table[2].C7_coords[2] += IP_table[i].C7_coords[2];
      ++CR_counter;
      }
    }
  //Determine if centroid atoms exist (CR_counter!=0):
  if(CR_counter==0)  {
    strncpy(OP_table[2].C1_name, "PSEUDO", 6);
    strncpy(OP_table[2].C3_element, " CR ", 4);
    strncpy(OP_table[2].C4_aminoacid, IP_table[0].C4_aminoacid, 3);
    strncpy(OP_table[2].C12_element, " X", 2);
    }
  else  {
    //Determine mean centroid co-ordinates:
    OP_table[2].C7_coords[0] /= CR_counter;
    OP_table[2].C7_coords[1] /= CR_counter;
    OP_table[2].C7_coords[2] /= CR_counter;

    //Write other relevant centroid values:
    strncpy(OP_table[2].C1_name, "ATOM  ", 6);
    strncpy(OP_table[2].C3_element, " CR ", 4);
    OP_table[2].C_altloc = altloc_flag;
    strncpy(OP_table[2].C4_aminoacid, IP_table[0].C4_aminoacid, 3);
    OP_table[2].C4_short = IP_table[0].C4_short;
    OP_table[2].C5_chain = IP_table[0].C5_chain;
    OP_table[2].C6_residue = IP_table[0].C6_residue;
    strncpy(OP_table[2].C12_element, " R", 2);
    }
  }

//Debugging: Print 'CA', 'CB', 'CR' lines:
//Flip the 0/1 'DEBUG' definition for easy debugging;
if(DEBUG==1)  {
  OP_table[0].put_line();
  OP_table[1].put_line();
  OP_table[2].put_line();
  }
}

//FUNCTION: Remove 'alt_loc' multiple-atoms. Leave only the first atom in place.
void no_altloc(PDBclass *IP_PDBtable, int *no_lines)
{
//Allocate memory for output-lines:
int OP_start=*no_lines;
PDBclass *OP_PDBtable=new PDBclass[*no_lines];

//Remove unwanted alternate-location atoms:
int i, j, same_flag;

for(i=*no_lines-1;i>=0;--i)  {
  same_flag = 0;
  for(j=i-1;j>=0;--j)  {
    if(IP_PDBtable[i].C6_residue==IP_PDBtable[j].C6_residue)  {
      if(strncmp(IP_PDBtable[i].C3_element, IP_PDBtable[j].C3_element, 4)==0)  {
        if(IP_PDBtable[i].C5_chain==IP_PDBtable[j].C5_chain)  {
          same_flag = 1;
          break;
          }
        }
      }
    }
  if(same_flag==0)  {
    --OP_start;
    OP_PDBtable[OP_start] = IP_PDBtable[i];
    }
  }

//Copy 'OP_PDBtable' into 'IP_PDBtable':
int IP_start=0;
for(i=OP_start;i<*no_lines;++i)  {
  IP_PDBtable[IP_start] = OP_PDBtable[i];
  ++IP_start;
  }
*no_lines = IP_start;

}















































