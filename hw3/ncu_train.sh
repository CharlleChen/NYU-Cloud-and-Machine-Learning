# run_all
for i in {10..100..10}
do
(ncu --log-file logs/$i --profile-from-start off --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum --target-processes all python3 ncu_main.py --dry-run --batch-size $i --epoch 1 --log-interval 10 )
done

# increase cnn
for i in {10..100..10}
do
(ncu --log-file logs2/$i --profile-from-start off --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum --target-processes all python3 ncu_main.py --dry-run --batch-size $i --epoch 1 --log-interval 1 --inc-conv)
done

# increase linear
for i in {10..100..10}
do
(ncu --log-file logs3/$i --profile-from-start off --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum --target-processes all python3 ncu_main.py --dry-run --batch-size $i --epoch 1 --log-interval 1 --inc-linear)
done
