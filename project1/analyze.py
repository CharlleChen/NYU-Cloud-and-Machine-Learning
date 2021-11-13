import os

MODEL=['resnet18', 'resnet34', 'resnet50']
PATH='v100'


def calculate_flops(filename):
  flops = 0
  with open(filename, 'r') as f:
    for i in f.readlines():
      if 'add' in i or 'mul' in i:
        flops += int(i.split()[-1])
      elif 'fma' in i:
        flops += 2 * int(i.split()[-1])
  return flops

def calculate_mem(filename):
  mem = 0
  with open(filename, 'r') as f:
    for i in f.readlines():
      if 'sectors' in i:
        mem += int(i.split()[-1])
  return mem * 32

def calculate_time(filename):
  with open(filename, 'r') as f:
    return int(f.read().strip().split()[-1]) / 1e9

with open(f'{PATH}.txt', 'w') as f:
  for i, m in enumerate(MODEL):

    print(m)
    ncu_ai = os.path.join(PATH, f'{m}-ncu-AI.log')
    ncu_runtime = os.path.join(PATH, f'{m}-nsys-runtime.log')

    flops = calculate_flops(ncu_ai)
    mem = calculate_mem(ncu_ai)
    time = calculate_time(ncu_runtime)
    # The gpu time measured by nvprof
    # if PATH=='v100':
    #   time = [i / 1e6 for i in [66227, 76324, 203248]][i]
    print(time, flops, mem)
    print("AI flop per byte", flops/mem)
    print('TFLOP/S', flops/time/1e12) 
    f.write(f'{m} {flops/mem} {flops/time/1e12}\n')