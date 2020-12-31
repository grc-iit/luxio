#! /usr/bin/env bash


CWD=$(pwd)
ORANGEFS_STUFF="$CWD"
ORANGEFS_CONFS="$CWD/conf/"
HACC_LOC="$HOME/hacc-io/"
DL="$CWD/darshan_logs"
HACC_VAL=4194304
#HACC_VAL=16777216
MPI_PROCS=40
TIER="nvme"
PVFS_LOC="/mnt/$TIER/nrajesh/orangefs/"

set -e
cd $ORANGEFS_STUFF
for i in conf/*; do
	bash deploy.sh $i $TIER

	cd  $HACC_LOC

	echo "############################################################################ $i"
	echo "############################ Doing WRITE $i########################################"
	echo "############################################################################ $i"
	mpirun -n $MPI_PROCS  ./hacc_io_write $HACC_VAL  $PVFS_LOC/file.out
	mv -vf  $DL/*hacc_io_write*.darshan ~/the_hacc_io_write_$i/
	rm -f $DL/*.darshan


	echo "########################################################################### $i"
	echo "################### Doing READ $i##################################################"
	echo "############################################################################ $i"
	mpirun -n $MPI_PROCS  ./hacc_io_read $HACC_VAL $PVFS_LOC/file.out
	mv -vf  $DL/*hacc_io_read*.darshan ~/the_hacc_io_read_$i/
	rm -f $DL/*.darshan
#	rm /mnt/nvme/nrajesh/write/file.out

	cd  $ORANGEFS_STUFF
	bash stop_all.sh 
done
