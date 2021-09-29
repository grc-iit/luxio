#! /usr/bin/env bash
set -e
set -u
set -x

if [ $# -eq 0 ]
then
        echo "bash md_test.sh <num procs> <num of files> <directory> <client file>"
        exit 0
fi

PROCS=$1
NUMFILES=$2
DIR=$3
CLIENT_FILE=$4
# RANK=$(python3 -c "from mpi4py import MPI;comm = MPI.COMM_WORLD;print(comm.Get_rank())")
# cd $DIR
# mkdir -p metadata_test_folder
echo "######################### CREATE ################################################################"
mpirun --hostfile $CLIENT_FILE  -n $PROCS mdtest -C -I $NUMFILES -d $DIR # | awk '$2 ~/creation/ {print $1 $2, $4}'
echo "######################## REMOVE #################################################################"
mpirun --hostfile $CLIENT_FILE  -n $PROCS mdtest -r -I $NUMFILES -d $DIR # | awk '$2 ~/removal/  {print $1 $2, $4}'
# rm -rf metadata_test_folder



# PROCS=$1
# NUMFILES=$2
# # RANK=$(python3 -c "from mpi4py import MPI;comm = MPI.COMM_WORLD;print(comm.Get_rank())")
#
# mkdir -p metadata_test_folder
# CREATE=$(mpirun -n $PROCS mdtest -C -I $NUMFILES -d metadata_test_folder)
# FILECREATE=$(echo $CREATE | awk '$1 ~ /File/ && $2 ~/creation/ {print $4}')
# DIRCREATE=$(echo $CREATE | awk '$1 ~ /Directory/ && $2 ~/creation/ {print $4}')
# echo "FileCreate $FILECREATE"
# echo "DirCreate $DIRCREATE"
#
# REMOVE=$(mpirun -n $PROCS mdtest -C -I $NUMFILES -d metadata_test_folder)
# FILEREMOVE=$(echo $CREATE | awk '$1 ~ /File/ && $2 ~/creation/ {print $4}')
# DIRREMOVE=$(echo $CREATE | awk '$1 ~ /Directory/ && $2 ~/creation/ {print $4}')
# REMOVE=$(mpirun -n $PROCS mdtest -r -I $NUMFILES -d metadata_test_folder  | awk '$1 ~ /File/ && $2 ~/removal/ {print $4}')
# echo "FileRemove $FILEREMOVE"
# echo "DirRemove $DIRREMOVE"
#
# rm -rf metadata_test_folder
