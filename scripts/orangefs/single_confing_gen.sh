#! /usr/bin/env bash

if [ $# -eq 0 ]
then
        echo "bash single_config_gen.sh <base config template>"
        exit 1
fi


source orangefs_ranges.conf

BASECONF=$1
FILE=orangefs_ranges.conf
TMPFILE=$(tempfile -d ./)
CWD=$(pwd)
for i in $(seq 0 $(expr ${#Variables[@]} - 1 ) )
do
        k=$(echo ${Variables[$i]} | awk '{split($1,a,"_"); print a[1]}')
        l=${Variables[$i]}
        eval vararray=\( \${${l}[@]} \)
        for val in ${vararray[@]}
        do
                sed "s/$k [A-Za-z0-9,]*/$k ${val}/g" $CWD/$BASECONF > $CWD/single_change_confs/${k}_${val}_orangefs.conf
                echo "$CWD/single_change_confs/${k}_${val}_orangefs.conf"

        done
done
