#! /usr/bin/env bash

set -u
set -e
set -x

CWD=$(pwd)
ORANGEFS_STUFF="$CWD"
# ORANGEFS_CONFS="$CWD/conf/"
ORANGEFS_CONFS="$CWD/confs"
DL="$CWD/darshan_logs"
HACC_VAL=4194304
#HACC_VAL=16777216
MPI_PROCS=40
TIER="nvme"
PVFS_LOC="/mnt/nvme/nrajesh/write/generic"
CLIENTS=( 1 4 8 16 )
SERVERS=( 4 8 16 )
IOROPTIONS=( "" "-z" )
IOSIZES=( "16mb" "1mb" "1kb" )
DSIZE="512m"
MDFILES=10000
SNO=1
FOLDER_INDEX=$(date +%Y%m%d%H%M%S)
CSV_HEADERS="$CWD/csv_stuff-$FOLDER_INDEX"

mkdir -p $CSV_HEADERS

# s/n clients servers network device storage config req_size mdm_reqs_per_process IO_type Total_Size mdm_time write_time read_time
## TODO device: add logic to change the DataStorageSpace and MetadataStorageSpace between nvme, ssd and hdd (change the val in the config)

for CI in "${CLIENTS[@]}"
do
	for SI in "${SERVERS[@]}"
	do

		for CONF in "$ORANGEFS_CONFS"_"$SI"/*
		do
			cd $ORANGEFS_STUFF
			bash deploy.sh "$CONF" "$CWD"/hostfiles/hostfile_servers_"$SI" "$CWD"/hostfiles/hostfile_clients_"$CI"

			cd $PVFS_LOC

			for io in "${IOSIZES[@]}"
			do
				for ioop in "${IOROPTIONS[@]}"
				do
					# add another look for TODO device
					echo $SNO >> $CSV_HEADERS/sno
					echo $SI >> $CSV_HEADERS/servers
					echo "SSD" >> $CSV_HEADERS/device
					echo "PFS" >> $CSV_HEADERS/storage
					echo $CONF >> $CSV_HEADERS/confs
					echo $io >> $CSV_HEADERS/req_size
					# set +e

					echo $CI >> $CSV_HEADERS/clients
					if [ "$ioop" == "" ]
					then
						echo "sequential" >> $CSV_HEADERS/iotype
					else
						echo "random" >> $CSV_HEADERS/iotype
					fi
					echo $DSIZE >> $CSV_HEADERS/total_size

					echo $MDFILES >> $CSV_HEADERS/mdmrpp

					echo "############################ Doing MDTEST $CONF########################################"
					bash $CWD/md_test.sh $(( $MPI_PROCS * $CI )) $MDFILES $PVFS_LOC "$CWD"/hostfiles/hostfile_clients_"$CI" >> $CSV_HEADERS/md_test
					# mpirun --hostfile ${CWD}/hostfiles/hostfile_clients_${CI} -n $(( $MPI_PROCS * $CI  )) mdtest -C -I $MDFILES -d $PVFS_LOC | awk '$2 ~/creation/ {print $1,$2,$4}'
					# mpirun --hostfile ${CWD}/hostfiles/hostfile_clients_${CI} -n $(( $MPI_PROCS * $CI  )) mdtest -r -I $MDFILES -d $PVFS_LOC | awk '$2 ~/removal/  {print $1,$2,$4}'

					echo "############################ Doing IOR $ $CONF########################################"
					mpirun --hostfile  ${CWD}/hostfiles/hostfile_clients_${CI} -n $(( $MPI_PROCS * $CI )) ior -t $io -b $DSIZE $ioop >> $CSV_HEADERS/ior_op

				done
				SNO=$(( $SNO + 1 ))

			done

# set -e
			cd  $ORANGEFS_STUFF
			bash stop_all.sh "$CONF" "$CWD"/hostfiles/hostfile_servers_"$SI" "$CWD"/hostfiles/hostfile_clients_"$CI"
		done
	done
done

# cd  $ORANGEFS_STUFF
# bash stop_all.sh
