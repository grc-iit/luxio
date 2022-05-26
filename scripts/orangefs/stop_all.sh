#!/bin/bash
set -e
set -u

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


client_partition=comp
client_dir=/mnt/nvme/nrajesh/write/generic
server_loc=$2
server_list=( $(cat $server_loc) )
client_loc=$3
client_list=( $(cat $client_loc) )


#Config PFS
name="orangefs" #TODO: Allow renaming
comm_port=3334  #TODO: Allow changing

set +e
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
