#! /usr/bin/env bash

source orangefs_ranges.conf
for i in $(seq 0 $(expr ${#Variables[@]} - 1 ) )
do
	k=$(echo ${Variables[$i]} | awk '{split($1,a,"_"); print a[1]}')
	l=${Variables[$i]}
	eval vararray=\( \${${l}[@]} \)
	for val in ${vararray[@]}
	do
		echo "$k $val"
		sed "s/$k [A-Za-z0-9,]*/$k ${val}/g" orangefs_default.conf > confs/${k}_${val}_orangefs.conf
		echo "confs/${k}_${val}_orangefs.conf"

	done
done
