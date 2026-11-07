#!/bin/bash
# generalized sesquitruncation with order-5 vertices: gen3_run.sh from to
cd "$(dirname "$0")"
gcc -O2 -o gen3_filter gen3_filter.c -lm || exit 1
mkdir -p gen3
for n in $(seq $1 $2); do
  t0=$(date +%s)
  nice -n 10 ${PLANTRI:-plantri} -pc3f5 $n 2>/dev/null | ./gen3_filter > gen3/n$n.pc 2> gen3/n$n.err
  echo "$n $(cat gen3/n$n.err) $(( $(date +%s) - t0 ))s"
done
