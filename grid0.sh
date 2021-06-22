#!/bin/sh

for lr in 0.001 0.002 0.0001
do
  for do in 0.4 0.15
  do
	  for ba in 64 128 256
	  do
         	python main.py -s 0 --gpu 0 -x -a -le -ld -b $ba -l $lr -do $do
	  done
	done
done
