import matplotlib.pyplot as plt

PEAK_GFLOPS=15.7 * 1e3
PEAK_BAND_GBS=900
FLAT = 12
FILE = 'a100.txt'
LABEL = ['flop byte ratio (flop / byte)', 'Attainable Gflop/s']

ceiling_points = [[1,PEAK_BAND_GBS], 
                  [PEAK_GFLOPS/PEAK_BAND_GBS, PEAK_GFLOPS], 
                  [PEAK_GFLOPS/PEAK_BAND_GBS * FLAT, PEAK_GFLOPS]]

X = [i[0] for i in ceiling_points]
Y = [i[1] for i in ceiling_points]



data_point = []
labels = []
with open(FILE, 'r') as f:
  for line in f.readlines():
    label, x, y = line.split()
    
    labels.append(label)
    data_point.append([float(x), float(y) * 1e3]) # Gflop -> Tflop

pointX = [i[0] for i in data_point]
pointY = [i[1] for i in data_point]

fig = plt.figure(1,figsize=(10.67,6.6))
plt.clf()
ax = fig.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Arithmetic Intensity [FLOPs/Byte]')
ax.set_ylabel('Performance [GFLOP/sec]')

nx   = 10000
xmin = -1
xmax = 3
ymin = 1
ymax = 200000

ax.set_xlim(10**xmin, 10**xmax)
ax.set_ylim(ymin, ymax)

for i, data in enumerate(data_point):
  ax.scatter(*data, label=labels[i])
ax.legend()
ax.plot(X, Y, label='roofline')
# ax.scatter(pointX, pointY)
# for i, txt in enumerate(labels):
#   ax.annotate(txt, data_point[i], xycoords='figure points')
#   break

# plt.xlabel(LABEL[0])
# plt.ylabel(LABEL[1])
# plt.ylim(0)
# plt.xlim(0)
plt.show()