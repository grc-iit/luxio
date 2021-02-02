#! /usr/bin/env bash
# set -e

if [ $# -eq 0 ]
then
        echo "bash config_gen.sh <base config template>"
        exit 1
fi


source orangefs_ranges.conf

FILE=$1
# FILE=orangefs_default.conf
mkdir -p ./tmp_confs
mkdir -p ./confs
CONF_FOLDER="./confs"
# TMPFILE=$(tempfile -d ./tmp_confs)
TMPFILE=$(mktemp -p ./tmp_confs/ --suffix=.conf)
cat $FILE > $TMPFILE

for i in $(seq 0 $(expr ${#Variables[@]} - 1 ) )
do
        for conf in ./tmp_confs/*
        do
                k=$(echo ${Variables[$i]} | awk '{split($1,a,"_"); print a[1]}')
                l=${Variables[$i]}
                eval vararray=\( \${${l}[@]} \)
                for val in ${vararray[@]}
                do
                        TMPFILE=$(mktemp -p ./tmp_confs/ --suffix=.conf)
                        echo "$TMPFILE $k $val"
                        sed "s/$k [A-Za-z0-9,]*/$k ${val}/g" $conf > $TMPFILE

                done
                md5sum ./tmp_confs/* | sort | awk 'BEGIN{lasthash = ""} $1 == lasthash {print $2} {lasthash = $1}' | xargs rm -v
        done
        # md5sum ./tmp_confs/* | sort | awk 'BEGIN{lasthash = ""} $1 == lasthash {print $2} {lasthash = $1}' | xargs rm -v
done


j=1
for i in ./tmp_confs/*
do
        mv -v "$i" "$CONF_FOLDER/orangefs_$j.conf"
        j=$(( $j + 1 ))
done

rm -rfv ./tmp_confs
