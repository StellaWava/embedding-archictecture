#!/bin/bash

for latency in 0 5 10 20
do
  for loss in 0 0.001 0.01
  do
    for bw in 1000 100 10
    do
      python3 run_expl.py \
        --latency $latency \
        --jitter 5 \
        --loss $loss \
        --bandwidth $bw \
        --arch hnsw

      python3 run_expl.py \
        --latency $latency \
        --jitter 5 \
        --loss $loss \
        --bandwidth $bw \
        --arch ivf
    done
  done
done