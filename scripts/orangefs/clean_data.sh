#! /usr/bin/env bash

READ_FILE=$(mktemp -p ./ --suffix=.tmp)
WRITE_FILE=$(mktemp -p ./ --suffix=.tmp)
CONF_FILE=$(mktemp -p ./ --suffix=.tmp)

awk '$1 ~/#/ {n=split($4,a,"/"); split(a[n],b,"."); print b[1]}' $1 > $CONF_FILE
awk '$1 ~/read/ {print $2}' $1 | awk 'NR % 2' > $READ_FILE
awk '$1 ~/write/ {print $2}' $1 | awk 'NR % 2' > $WRITE_FILE

echo "READ: $(wc -l $READ_FILE)"
echo "WRITE: $(wc -l $WRITE_FILE)"
echo "CONF: $(wc -l $CONF_FILE)"


paste -d "," $CONF_FILE $READ_FILE $WRITE_FILE

rm $READ_FILE $CONF_FILE $WRITE_FILE
