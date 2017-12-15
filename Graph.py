import matplotlib.pyplot as plt
from configuration import configure
import numpy as np



cfg = configure("winequality-white.csv")

ds=cfg.getdataset()

#box plot
ds.plot(kind='box', subplots=False, layout=(5,3), legend = False, figsize = (30,15), table=ds.describe())
ax1 = plt.axes()
x_axis = ax1.axes.get_xaxis()
x_axis.set_visible(False)
x_label = x_axis.get_label()
x_label.set_visible(False)
plt.show()
plt.savefig("Charts/"+ "BoxwithTable" + "_plot_scores.jpg")

#correlation matrix
ds_corr = ds.corr()
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
cax = ax.matshow(ds_corr)
fig.colorbar(cax)
ticks = np.arange(0,12,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(list(ds),rotation=60)
ax.set_yticklabels(list(ds))
plt.show()
