for m in resnet18 resnet34 resnet50
do
nvprof -f --log-file $m.runtime --print-gpu-trace python3 main.py data --arch $m --epochs 1 --batch-size 10 --d 10 --seed 2021

echo $m

tail -n +5 $m.runtime | sed -e "s/us//g" | awk 'BEGIN{sum=0}{sum=sum+$2}END{printf("%d\n",sum);}'
done