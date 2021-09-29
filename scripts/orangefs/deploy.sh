#!/bin/bash
set -u
set -e
#Remeber to have env Variables for:
# ORANGEFS_KO
# ORANGEFS_PATH
# PVFS2TAB_FILE

CWD=$(pwd)

if [ $# -eq 0 ]
then
	echo "bash deploy.sh <conf file> <server list> <client list>"
	exit 0
fi
#Input Variables
server_partition=stor
num_servers=4
conf_file=$1
server_dir=$(awk '$1 ~ /DataStorageSpace/ {print $2}' $conf_file )
meta_dir=$(awk '$1 ~ /MetadataStorageSpace/ {print $2}' $conf_file )

#General Variables
server_loc=$2
server_list=( $(cat $server_loc) )
client_loc=$3
client_list=( $(cat $client_loc) )

client_partition=comp
client_dir="/mnt/nvme/nrajesh/write/generic"

#Config PFS
name="orangefs" #TODO: Allow renaming
comm_port=3334  #TODO: Allow changing

echo $CWD

# cat ${server_loc} | parallel -I% --max-args=1 '
# echo "Doing %"
# ssh % /bin/bash
# rm -rf ${server_dir}*
# rm -rf ${meta_dir}*
# mkdir -p ${server_dir}
# mkdir -p ${meta_dir}
# pvfs2-server -f -a % ${conf_file}
# pvfs2-server -a % ${conf_file}
# '
#Server Setup

set +e
for node in ${server_list[@]}
do
	ssh ${node} /bin/bash << EOF
echo "Setting up server at ${node}"
rm -rf ${server_dir}
rm -rf ${meta_dir}
mkdir -p ${server_dir}
mkdir -p ${meta_dir}
pvfs2-server -f -a ${node} ${conf_file}
pvfs2-server -a ${node} ${conf_file}
EOF
done

for node in ${client_list[@]}
do
	ssh ${node} /bin/bash << EOF
echo "Starting client on ${node}"
sudo kill-pvfs2-client
mkdir -p ${client_dir}
sudo insmod ${ORANGEFS_KO}/pvfs2.ko
sudo ${ORANGEFS_PATH}/sbin/pvfs2-client -p ${ORANGEFS_PATH}/sbin/pvfs2-client-core
echo "executing sudo mount -t pvfs2 tcp://${server_list[0]}:${comm_port}/${name} ${client_dir}"
sudo mount -t pvfs2 tcp://${server_list[0]}:${comm_port}/${name} ${client_dir}
EOF
done

# #Client Setup
# # for node in ${client_list[@]}
# # do
# cat ${client_loc} | parallel -I% --max-args=1 "ssh % /bin/bash
# echo "Starting client on %"
# sudo kill-pvfs2-client
# mkdir -p ${client_dir}
# sudo insmod ${ORANGEFS_KO}/pvfs2.ko
# sudo ${ORANGEFS_PATH}/sbin/pvfs2-client -p ${ORANGEFS_PATH}/sbin/pvfs2-client-core
# sudo mount -t pvfs2 tcp://${server_list[0]}:${comm_port}/${name} ${client_dir}
# "
# done
