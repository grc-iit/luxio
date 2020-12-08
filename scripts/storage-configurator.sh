#!/bin/bash
if [[ "$#" -lt 3 ]]; then
    ${LUXIOMAINDIR}/luxio.py --help
elif [[ "$#" -eq 3 ]]; then
    export tempout=`mktemp`
    ${LUXIOMAINDIR}/luxio.py -m 'conf' -j $1 -i $2 -s $3 -c "../sample/stor_req_conf_output.json" -o $tempout
    ./print-luxio.sh $tempout
    rm $tempout
elif [[ "$#" -eq 4 ]]; then
    ${LUXIOMAINDIR}/luxio.py -m 'conf' -j $1 -i $2 -s $3 -c "../sample/stor_req_conf_output.json" -o $4
else
    ${LUXIOMAINDIR}/luxio.py --help
fi
