import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

reg_channels = np.array([64, 128, 256, 512, 512])
pca_channels = np.array(
    [
        [23, 27, 19, 17, 24],
        [86, 86, 64, 68, 83],
        [179, 180, 163, 156, 173],
        [326, 269, 215, 220, 268],
        [222, 113, 110, 104, 119],
    ]
)
ind = np.array(range(5))
layers = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]
width = 0.35

fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind + width, reg_channels, width, color="royalblue", label="No PCA")
rects2 = ax.bar(
    ind,
    np.mean(pca_channels, axis=-1),
    width,
    color="seagreen",
    yerr=np.std(pca_channels, axis=-1),
    label="PCA",
)

for axis in [rects1, rects2]:
    for bar in axis:
        w, h = bar.get_width(), bar.get_height()
        plt.text(bar.get_x() + w / 2, bar.get_y() + h / 2, f"{h:.0f}", ha="center", va="center")

ax.set_ylabel("Number of channels in layer")
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(layers)
ax.legend((rects1[0], rects2[0], Line2D([0], [0], color="k", linewidth=1)), ("No PCA", "PCA", "Std. dev."))
plt.tight_layout()
plt.savefig("pca-gains.jpg")

data = pd.read_csv("perf.csv")
data["lin"] = data["optex-lin"].apply(
    lambda x: float(x.split(":")[0]) * 60 + float(x.split(":")[1]) if ":" in x else -1
)
data["cdf"] = data["optex-cdf"].apply(
    lambda x: float(x.split(":")[0]) * 60 + float(x.split(":")[1]) if ":" in x else -1
)
data.maua = data.maua.apply(lambda x: float(x.split(":")[0]) * 60 + float(x.split(":")[1]) if ":" in x else -1)
data.embark = data.embark.apply(lambda x: float(x.split(":")[0]) * 60 + float(x.split(":")[1]) if ":" in x else -1)
print(data)

fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(8, 6))
task = data[data["task"] == "synth"]

pixels = task["size"].values * task["size"].values

ax.plot(
    pixels[np.argwhere(task.lin.values > 0)],
    task.lin.values[np.argwhere(task.lin.values > 0)],
    label="optex linear hist",
)
ax.plot(
    pixels[np.argwhere(task.cdf.values > 0)], task.cdf.values[np.argwhere(task.cdf.values > 0)], label="optex cdf hist"
)
ax.plot(pixels[np.argwhere(task.maua.values > 0)], task.maua.values[np.argwhere(task.maua.values > 0)], label="maua")
ax.plot(
    pixels[np.argwhere(task.embark.values > 0)],
    task.embark.values[np.argwhere(task.embark.values > 0)],
    label="embark",
)

ax.set_title("execution time scaling")
ax.set_ylabel("seconds")
ax.set_xlabel("pixels")
plt.legend()
plt.tight_layout()
plt.savefig("perf.png")
