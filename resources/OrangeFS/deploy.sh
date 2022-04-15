#!/bin/bash

#Remeber to have env Variables for:
# ORANGEFS_KO
# ORANGEFS_PATH
# PVFS2TAB_FILE

CWD=$(pwd)

#Input Variables
server_partition=stor
num_servers=4
server_dir=/mnt/hdd/jcernudagarcia/orangefs

client_partition=comp
client_dir=/mnt/nvme/jcernudagarcia/write/generic

conf_file=${1}

#General Variables
client_list=($(cat ${CWD}/hostfiles/hostfile_clients))
server_list=($(cat ${CWD}/hostfiles/hostfile_servers))

#Config PFS
name="orangefs" #TODO: Allow renaming
comm_port=3334  #TODO: Allow changing

#Server Setup
for node in ${server_list[@]}
do
ssh ${node} /bin/bash << EOF
echo "Setting up server at ${node} "
rm -rf ${server_dir}*
mkdir -p ${server_dir}
pvfs2-server -f -a ${node} ${conf_file}
pvfs2-server -a ${node} ${conf_file}
EOF
done

#Client Setup
for node in ${client_list[@]}
do
ssh ${node} /bin/bash << EOF
echo "Starting client on ${node}"
sudo kill-pvfs2-client
mkdir -p ${client_dir} 
sudo insmod ${ORANGEFS_KO}/pvfs2.ko
sudo ${ORANGEFS_PATH}/sbin/pvfs2-client -p ${ORANGEFS_PATH}/sbin/pvfs2-client-core
sudo mount -t pvfs2 tcp://${server_list[0]}:${comm_port}/${name} ${client_dir}
EOF
done
