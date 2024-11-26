#!/bin/bash
#Step0-cabbage.sh: This wrapper script contains multiple options for creating/handling cabbage-files:

#Standard error message (general):
if [[ $# -ne 1 ]]
then
	echo "usage: Step0-cabbage.sh <Input file/directory>"
	exit
fi

#pre-execution cleanup:
rm outfile.cabbage

anchor=`pwd`
cd "$1"
address=`pwd`
cd $anchor
#Convert all <pocket.pdb files> into individual <cabbage unit files>:
#Concatenate <cabbage unit files> into <cabbage datafile>:
for i in `ls $address`
do
	./Step0-cabbage_core $address/$i >> outfile.cabbage
done
echo "Here"
#Insert END-OF-FILE string into <cabbage datafile>:
./Step0-END-FILE >> outfile.cabbage


  












