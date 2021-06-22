# GPU 3
for lr in 0.001 0.002 0.0001
do
	for ba in 16 32 64 128 256
	do
       	python main.py -s 3 --gpu 3 -x -a -le -ld -b $ba -lr $lr
	done
done
