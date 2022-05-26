#! /usr/bin/env bash
set -u
set -e
# set -x

if [ $# -lt 2 ]
then 
  echo "$0 <darshan file> <output_file>"
  exit 1
fi
CWD=$(pwd)
DARSHAN_FILE=$1
OUTFILE=$2
TMPFILE=$(mktemp -p $CWD)
TMPSERIES=$(mktemp -p $CWD)
TMPDUR=$(mktemp -p $CWD)

# echo "Start,End,Duration,Offset,Length"
darshan-dxt-parser $DARSHAN_FILE | awk '$1 ~ /X_/ {print $7, $8, $8-$7, $5, $6}' | sort -k 1 -n  > $TMPFILE
cat $TMPFILE | awk '{print $1, "+"$5}' > $TMPSERIES
cat $TMPFILE | awk '{print $2, "-"$5}' >> $TMPSERIES
cat $TMPSERIES | sort -k 1 -n | awk '{total += $2; print $0, total}' | awk 'BEGIN{total=0}{if ($2 > 0) {total += 1} else {total-=1}; print $1"," $2"," $3"," total}' > $TMPFILE
cat $TMPFILE | awk -F ',' 'NR > 1 {print $1-p}{p=$1} END {print 0}' > $TMPDUR
echo "timestamp,delta_data,total_io_size,num_io_ops,duration" > $OUTFILE
paste -d ',' $TMPFILE $TMPDUR >> $OUTFILE

rm $TMPFILE
rm $TMPSERIES
rm $TMPDUR
