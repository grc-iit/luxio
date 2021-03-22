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

#Stop clients
for node in ${client_list[@]}
do
ssh ${node} /bin/bash << EOF
echo "Stopping client on $node"
sudo /usr/sbin/kill-pvfs2-client
EOF
done

#Stop servers
for node in ${server_list[@]}
do
ssh ${node} /bin/bash << EOF
echo "Killing server at ${node} "
sudo /usr/sbin/kill-pvfs2-client
rm -rf ${server_dir}/*
killall -s SIGKILL pvfs2-server
EOF
done

echo "done"
