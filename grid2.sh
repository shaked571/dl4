#!/bin/sh
# GPU 2

for lr in 0.001 0.002 0.003
do
  for drop in 0.3 0.25 0.2
  do
    python main.py -s 2 --gpu 2 -x -a -le -ld -l $lr -do $drop -r -e 35
	done
done
