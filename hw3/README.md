# Cloud and Machine Learning Assignment 3
Profiling neural networks on GPU

> The training script (contains `main.py`) is adapted from [pytorch exampes](https://github.com/pytorch/examples)

## File Description
#### Model evaluation
- `flop_counter.py`: evaluate model FLOPs and model parameters before training

#### NVIDIA Nsight tool
- `ncu_main.py`: code to train the neural network with NCU, a NVIDIA profiling tool
- `ncu_train.sh`: the script to run `ncu_main.py` in batch (with different batch size)
- `ncu_log.zip`: the logs files get from ncu
- `generate_report.sh`: generate the report (sum up all the operations) from the log files
- `plot.py`: plot the graph (FLOPs versus batch size)

#### Pytorch Profiler
- `profiler_main.py`: code to profile the neural network with pytorch profiler
- `run_pytorch_profiler.sh`: the script to run `profiler_main.py` in batch (with different batch size)
- `profiler_log.zip`: the logs files get from Pytorch Profiler, recommended to be open with tensorboard
