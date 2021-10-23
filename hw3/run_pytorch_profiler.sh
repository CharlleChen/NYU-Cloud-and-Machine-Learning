#!bin/sh

i=20

while [ $i -le 100 ]
do
	python3 profiler_main.py --epoch 10 --batch-size $i
	i=`expr $i + 10`
done
