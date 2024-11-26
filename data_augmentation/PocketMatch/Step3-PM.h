/* Step3-PMserial.h:
Used to store data-structures and functions for 'Step3-PMserial.c'
*/

//FUNCTION: Create a hash-table:
twin_int *make_hash(int hash_size)
{
twin_int *hash_table;
//Allocate memory for hash-table:
hash_table = calloc((hash_size*hash_size +2*hash_size +1), sizeof(twin_int));

//Write 'hash_size' into table (first entry):
(hash_table)->i[0] = hash_size;
(hash_table)->i[1] = hash_size;

//Return allocated memory:
return hash_table;
}

//FUNCTION: Print entire hash table:
int print_hash(twin_int *hash_table)
{
//Extract sizeof hash-table from input, align pointer to start of hash-table:
int hash_size;
hash_size = (hash_table+0)->i[0];
++hash_table;

int i, j;
for(i=0;i<hash_size;++i)  {
  printf("key: %4d|", (hash_table+i*(hash_size+2))->i[0]);
  printf("max: %4d|", (hash_table+i*(hash_size+2)+1)->i[0]);
  for(j=0;j<hash_size;++j)  {
    printf(" %4d", (hash_table+i*(hash_size+2)+2+j)->i[0]);
    }
  printf("\n");
  printf("          ");
  printf("max: %4d|", (hash_table+i*(hash_size+2)+1)->i[1]); 
  for(j=0;j<hash_size;++j)  {
    printf(" %4d", (hash_table+i*(hash_size+2)+2+j)->i[1]);
    }
  printf("\n");
  }

return 0;
}

//FUNCTION: Input entries into allocated hash-table:
int IP_hash(twin_int *hash_table, int *IP_value)
{
//Extract sizeof hash-table from input, align pointer to start of hash-table:
int hash_size;
hash_size = (hash_table+0)->i[0];
++hash_table;

//Perpare hashing-key:
int turns1=0, turns2=0, turns3=0;
int address1, address2, address3;
int hash_key1, hash_key2, hash_key3;
hash_key1 = *(IP_value+0)%hash_size;
hash_key2 = *(IP_value+1)%hash_size;
hash_key3 = *(IP_value+2)%hash_size;

//Use double-hashing to write data into 'hash_table':
//FIRST HASH: Write 'IP_value+0' under hashing-key:
while(1)  {
  address1 = hash_key1*(hash_size+2);
  if((hash_table+address1)->i[0]==0 || (hash_table+address1)->i[0]==*(IP_value+0))  {
    (hash_table+address1)->i[0] = *(IP_value+0);
    //SECOND HASH #1: Write 'IP_value+1' (hashing-data) under 'IP_value+0':
    while(1)  {
      address2 = address1+2+hash_key2;
      if((hash_table+address2)->i[0]==0 || (hash_table+address2)->i[0]==*(IP_value+1))  {
        (hash_table+address2)->i[0] = *(IP_value+1);
        ++(hash_table+address2)->i[1];
        //Find maximum sub-hashed values for 'IP_value+1':
        if((hash_table+address2)->i[1] > (hash_table+address1+1)->i[1])  {
          (hash_table+address1+1)->i[0] = (hash_table+address2)->i[0];
          (hash_table+address1+1)->i[1] = (hash_table+address2)->i[1];
          }
        break;
        }
      else  {
        if(hash_key2<hash_size-1) ++hash_key2;
        else hash_key2=0;
        ++turns2;
        if(turns2==hash_size)  {
          printf("<Step3-PM_serial> Error: function 'IP_hash' suffered secondary #1 bin-overflow\n");
          return 1;
          }
        }
      }
    //SECOND HASH #2: Write 'IP_value+2' (hashing-data) under 'IP_value+0':
    while(1)  {
      address3 = address1+2+hash_key3;
      if((hash_table+address3)->i[0]==0 || (hash_table+address3)->i[0]==*(IP_value+2))  {
        (hash_table+address3)->i[0] = *(IP_value+2);
        ++(hash_table+address3)->i[1];
        //Find maximum sub-hashed values for 'IP_value+1':
        if((hash_table+address3)->i[1] > (hash_table+address1+1)->i[1])  {
          (hash_table+address1+1)->i[0] = (hash_table+address3)->i[0];
          (hash_table+address1+1)->i[1] = (hash_table+address3)->i[1];
          }
        break;
        }
      else  {
        if(hash_key3<hash_size-1) ++hash_key3;
        else hash_key3=0;
        ++turns3;
        if(turns3==hash_size)  {
          printf("<Step3-PM_serial> Error: function 'IP_hash' suffered secondary #2 bin-overflow\n");
          return 1;
          }
        }
      }
    //FIRST HASH: 'break' statememt:
    break;
    }
  else  {
    if(hash_key1<hash_size-1) ++hash_key1;
    else hash_key1=0;
    ++turns1;
    if(turns1==hash_size)  {
      printf("<Step3-PM_serial> Error: function 'IP_hash' suffered primary bin-overflow\n");
      return 1;
      }
    }
  }

return 0;
}

//FUNCTION: Compare, extract common a.a. residues from the hash-tables:
void compare_hash(twin_int *hash_table1, twin_int *hash_table2, int *SUMA_score, char *SUMA_block)
{
*SUMA_score = 0;
int i;

int value1, value2, address1, address2;
int hash_size1, hash_size2, hash_key, turns;

//Extract sizeof hash-tables from input, align pointer to start of hash-tables:
hash_size1 = (hash_table1+0)->i[0];
++hash_table1;
hash_size2 = (hash_table2+0)->i[0];
++hash_table2;
if(hash_size1!=hash_size2)  {
  printf("<Step3-PM_serial> Error: function 'compare_hash' was fed hash-tables of different sizes\n");
  printf("hash_table1 = %d bytes|hash_table2 = %d bytes\n", hash_size1, hash_size2);
  }

//Start comparing:
char compare[17];
bzero(compare, sizeof(compare));
for(i=0;i<hash_size1;++i)  {
  address1 = i*(hash_size1+2);
  value1 = (hash_table1+address1)->i[0];
  if(value1!=0)  {
    value2 = (hash_table1+address1+1)->i[0];
    hash_key = value2%hash_size1;
    //Find 'hash_key' in 'hash_table2':
    turns = 0;
    while(1)  {
      address2 = hash_key*(hash_size1+2);
      if((hash_table2+address2)->i[0]==value2 && (hash_table2+address2+1)->i[0]==value1)  {
        //A SUMA-match has been found:
        sprintf(compare, " %c-%.4d__%c-%.4d ", value1/CHAIN_FACTOR+'@', value1%CHAIN_FACTOR, value2/CHAIN_FACTOR+'@', value2%CHAIN_FACTOR);
        if(*SUMA_score<(BLOCK_SIZE-64)/(2*8*sizeof(char)))  {
          strncpy(SUMA_block+64+*SUMA_score*16, compare, 16);
          }
        ++*SUMA_score;
        break;
        }
      else  {
        ++turns;
        if(hash_key<hash_size1-1) ++hash_key;
        else hash_key=0;
        if(turns==hash_size1)  {
          break;
          }
        }
      }

    }
  }

//Fill unused 'SUMA_block' with " ?-0000__?-0000 " characters:
for(i=64+*SUMA_score*16;i<BLOCK_SIZE;i+=16*sizeof(char))  {
  strncpy(SUMA_block+i, " ?-0000__?-0000 ", 16);
  }

}








