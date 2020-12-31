#!/bin/bash

#Remeber to have env Variables for:
# ORANGEFS_KO
# ORANGEFS_PATH
# PVFS2TAB_FILE

CWD=$(pwd)

#Input Variables
server_partition=stor
num_servers=4
TIER=$2
server_dir=/mnt/$TIER/nrajesh/orangefs
meta_dir=/mnt/$TIER/nrajesh/meta

client_partition=comp
client_dir=/mnt/nvme/nrajesh/write/generic

conf_file=${1}

#General Variables
client_loc=${CWD}/hostfiles/hostfile_clients
client_list=($(cat ${CWD}/hostfiles/hostfile_clients))
server_loc=${CWD}/hostfiles/hostfile_servers
server_list=($(cat ${CWD}/hostfiles/hostfile_servers))

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
for node in ${server_list[@]}
do
ssh ${node} /bin/bash << EOF
echo "Setting up server at ${node}"
rm -rf ${server_dir}*
rm -rf ${meta_dir}*
mkdir -p ${server_dir}
mkdir -p ${meta_dir}
pvfs2-server -f -a ${node} ${conf_file}
pvfs2-server -a ${node} ${conf_file}
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
