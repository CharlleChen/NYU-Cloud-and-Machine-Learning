# Project 1: Roofline model
Performance is a crucial topic in chip industry. From the designer's point of view, aside from the clock frequency, there are many critical factors to consider to boost the actual performance. From the application level, programmers care about the performance because the utilization of CPU/GPU largely determine the development efficiency and cost. This report introduces a graphic representation tool  -- roofline model -- for programmers to understand the performance of their programs and goes through an experiment to see how the roofline model illustrates the performance on different GPU and convolutional nerual networks.

## Report
- `doc` folder

## Roofline model
Check definitions and basic concepts at WikiPedia

## Measurement
### Total flops, memory bandwidth, and runtime
- run_test.sh

Use `ncu` and `nsys` from Nsight Compute and Nsight System tools to profile flops & memory bandwidth and runtime respectively.
```bash
sh run_test.sh resnet18; sh run_test.sh resnet34; sh run_test.sh resnet50;
```

Alternatively, use `run_nvprof_runtime.sh`, which utilize `nvprof` to measure runtime
```bash
sh run_nvprof_runtime.sh
```

## Data 
- A100 folder stores the data from A100
- V100 folder stores the data from V100


## Analysis and plotting
change `PATH` to {V100, A100} in the `analyze.py`

Change `FILE` to {v100.txt, a100.txt} in the `plot.py`

```bash
python3 analyze.py

python3 plot.py
```
