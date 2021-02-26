#!/bin/bash
if [[ "$#" -lt 2 ]]; then
    ${LUXIOMAINDIR}/luxio.py --help
elif [[ "$#" -eq 2 ]]; then
    export tempout=`mktemp`
    ${LUXIOMAINDIR}/luxio.py -m 'io' -j $1 -t $2 -i "../sample/io_req_output.json" -s "../sample/stor_req_output.json" -c "../sample/stor_req_conf_output.json" -o $tempout
    ./print-luxio.sh $tempout
    rm $tempout
elif [[ "$#" -eq 3 ]]; then
    ${LUXIOMAINDIR}/luxio.py -m 'io' -j $1 -t $2 -i "../sample/io_req_output.json" -s "../sample/stor_req_output.json" -c "../sample/stor_req_conf_output.json" -o $3
else
    ${LUXIOMAINDIR}/luxio.py --help
fi
