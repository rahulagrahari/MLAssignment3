import matplotlib.pyplot as plt
# import matplotlib.axes.Axes as axes
from configuration import configure
import numpy as np
import datetime, time


cfg = configure("white")

ds=cfg.getdataset()

# #box plot
# ds.plot(kind='box', subplots=False, layout=(5,3), legend = False, figsize = (30,15), table=ds.describe())
# ax1 = plt.axes()
# x_axis = ax1.axes.get_xaxis()
# x_axis.set_visible(False)
# x_label = x_axis.get_label()
# x_label.set_visible(False)
# plt.show()
# plt.savefig("Charts/"+ "BoxwithTable" + "_plot_scores.jpg")

ds.diff().hist(color='k', alpha=0.5, bins=12, figsize = (15,15))
plt.show()
plt.close()
plt.savefig("Charts/"+ "datafreq" + "_plot_scores.jpg")
# ds.describe()


ds_corr = ds.corr()
print (ds_corr)
# fig = plt.figure(figsize=(15,15))
# ax = fig.add_subplot(111)
# cax = ax.matshow(ds_corr)
# fig.colorbar(cax)
# ticks = np.arange(0,12,1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(list(ds),rotation=60)
# ax.set_yticklabels(list(ds))

# ds.hist(figsize=(15,15))
# # plt.distplot(ds)
# plt.show()
# plt.savefig("Charts/"+ "datafreq" + "_plot_scores.jpg")
# y_axis=ds["quality"]
# x_axis=ds.drop("quality",axis=1)
# plt.bar(x_axis, y_axis, label="score", color='g')
# plt.legend()
# plt.xlabel("feature")
# plt.ylabel("Quality")
# plt.show()


# for feature in list(ds):
#     y_axis=ds["quality"]
#     x_axis=ds[feature]
#     # plt.scatter(x_axis, y_axis, label="score", color='b')
#     # plt.legend()
#     # plt.xlabel(feature)
#     # plt.ylabel("Quality")
#     st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
#     plt.savefig("Charts/"+ st + "_plot_scores.pdf")
#     plt.show()
