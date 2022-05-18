#!/bin/bash
if [[ "$#" -lt 1 ]]; then
    ${LUXIOMAINDIR}/luxio.py --help
elif [[ "$#" -eq 1 ]]; then
    export tempout=`mktemp`
    ${LUXIOMAINDIR}/luxio.py -m 'stor' -i $1 -s "../resources/stor_req_output.json" -o $tempout
    ./print-luxio.sh $tempout
    rm $tempout
elif [[ "$#" -eq 2 ]]; then
    ${LUXIOMAINDIR}/luxio.py -m 'stor' -i $1 -s "../resources/stor_req_output.json" -o $2
else
    ${LUXIOMAINDIR}/luxio.py --help
fi
