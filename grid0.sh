#!/bin/sh

# GPU 0
for lr in 0.001 0.002 0.0001
do
	for ba in 16 32 64 128 256
	do
       	python main.py -s 0 --gpu 0 -x -a -le -ld -b $ba -l $lr
	done
done
