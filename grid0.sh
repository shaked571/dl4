#!/bin/sh
# GPU 0

for lr in 0.001 0.002 0.0001
do
  for do in 0.4 0.15
  do
	  for ba in 64 128 256
	  do
	    for r in true false
	    do
	      for d in true false
	      do
	        if $d && $r; then
         	  python main.py -s 0 --gpu 0 -x -a -le -ld -b $ba -l $lr -do $do -d -r
         	elif $d; then
         	  python main.py -s 0 --gpu 0 -x -a -le -ld -b $ba -l $lr -do $do -d
          elif $r; then
         	  python main.py -s 0 --gpu 0 -x -a -le -ld -b $ba -l $lr -do $do -r
          else
         	  python main.py -s 0 --gpu 0 -x -a -le -ld -b $ba -l $lr -do $do
         	fi
         	done
        done
	  done
	done
done
