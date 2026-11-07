#!/bin/bash
# usage: gen2_run.sh n_from n_to [mod]  (plantri | C filter | python canon -> gen2/n<N>.txt)
cd "$(dirname "$0")" && mkdir -p gen2
[ -x gen2_filter ] || gcc -O2 -o gen2_filter gen2_filter.c -lm
for n in $(seq $1 $2); do
  t0=$(date +%s)
  nice -n 10 ${PLANTRI:-plantri} -pc3f4 $n 2>/dev/null | nice -n 10 ./gen2_filter 2> gen2/n$n.err | python3 gen2_plantri.py > gen2/n$n.txt 2>/dev/null
  echo "$n kept $(wc -l < gen2/n$n.txt) $(cat gen2/n$n.err) $(( $(date +%s) - t0 ))s"
done
