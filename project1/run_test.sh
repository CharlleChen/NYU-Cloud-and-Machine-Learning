if [ -z $1 ]
then
	echo "Must input model arch"
	exit 1
fi
MODEL=$1

RUN_COMMAND="python3 main.py data --arch $MODEL --epochs 1 --batch-size 10 --print-freq 10 --seed 2021"
NCU_METRICS="dram__sectors_write.sum,dram__bytes_write.sum.per_second,dram__sectors_read.sum,dram__bytes_read.sum.per_second,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
NCU_LOG_FILE="$MODEL-ncu-AI.log"
NCU_RAW_LOG="$MODEL-raw-ncu.log"
rm $NCU_LOG_FILE || true
rm $NCU_RAW_LOG || true

NSYS_RUNTIME_LOG="$MODEL-nsys-runtime.log"
NSYS_RAW_RUNTIME="$MODEL-nsys-raw.qdrep"
rm $NSYS_RAW_RUNTIME || true

echo "Testing on model: $MODEL"
echo "Run command: $RUN_COMMAND"


# Run
ncu -f --log-file $NCU_RAW_LOG --metrics $NCU_METRICS --target-processes all $RUN_COMMAND

for TYPE in ${NCU_METRICS//,/ }
do
cat $NCU_RAW_LOG | grep $TYPE | sed -e "s/,//g" | awk -v t="$TYPE" 'BEGIN{sum=0}{sum=sum+$3}END{printf("%s %d\n", t, sum);}' >> $NCU_LOG_FILE
done

# time test
nsys profile -f true -o $NSYS_RAW_RUNTIME $RUN_COMMAND
rm tmp_gputrace.csv || true # ignore error
nsys stats --report gputrace $NSYS_RAW_RUNTIME -o tmp
tail -n +2 tmp_gputrace.csv | sed -e "s/,/ /g" | awk -v t="Runtime(ns)" 'BEGIN{sum=0}{sum=sum+$2}END{printf("%s %d\n", t, sum);}' > $NSYS_RUNTIME_LOG

zip $MODEL.zip $NCU_LOG_FILE $NCU_RAW_LOG $NSYS_RUNTIME_LOG
exit 1