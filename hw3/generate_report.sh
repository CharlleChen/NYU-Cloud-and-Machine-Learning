cd logs

echo '' > report
for i in {10..100..10}                                                                   
do
file="${i}"
echo "${file}:" >> report

for TYPE in smsp__sass_thread_inst_executed_op_fadd_pred_on smsp__sass_thread_inst_executed_op_fmul_pred_on smsp__sass_thread_inst_executed_op_ffma_pred_on
do
cat $file | grep $TYPE | sed -e "s/,//g" | awk -v t="$TYPE" 'BEGIN{sum=0}{sum=sum+$3}END{printf("%s %d\n", t, sum);}' >> report
done

echo '' >> report

done
