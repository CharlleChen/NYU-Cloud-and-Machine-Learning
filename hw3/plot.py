import matplotlib.pyplot as plt
import os


def read(file_path):
    file_path = os.path.join(file_path, "report")
    x, y = [], []
    with open(file_path, "r") as f:
        lines = f.readlines()
    s = -1
    for line in lines:
        line = line.strip()
        if ":" in line:
            x.append(int(line[:-1]))
            if s != -1:
                y.append(s)
            s = 0
        elif line != "":
            if 'fma' in line:
              s += int(line.split()[-1]) * 2
            else:
              s += int(line.split()[-1])
    y.append(s)
    return x, y


def plot(x, y, legend):
    print(y)
    plt.scatter(x, y, label=legend)
    


def main():
    # key: path of log ; value: legend names
    files = {"logs": "base", "logs2": "ext_cnn", "logs3": "ext_fc"} # "logs4": "marked"
    fig, ax = plt.subplots()
    for f in files.keys():
        x, y = read(f)
        print(y)
        ax.scatter(x, y, label=files[f])

    ax.legend()

    plt.show()


main()

