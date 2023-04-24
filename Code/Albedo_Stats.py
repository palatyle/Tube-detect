import os

import data_wrangle as dw
import matplotlib.pyplot as plt
import numpy as np


working_directory = "C:\\Users\\palatyle\\Documents\\Tube-detect\\Albedo_calc"
os.chdir(working_directory)
working_list = os.listdir(working_directory)
array_list = []

for eachfile in working_list:
    each_gdal = dw.GDAL_read_tiff(eachfile)
    each_raster = dw.GDAL2NP(each_gdal)

    array_list.append(each_raster)

full_stack = np.stack(array_list, axis=0)
average = np.average(full_stack, axis=0)
std_dev = np.std(full_stack, axis=0)
print("Done!")

fig, ax = plt.subplots(2, 1)
img_plot = ax[0].imshow(std_dev)
fig.colorbar(img_plot, ax=ax[0])
img_plot2 = ax[1].imshow(average)
fig.colorbar(img_plot2, ax=ax[1])

ax[0].set_xlabel("x distance")
ax[0].set_ylabel("y diatance")
ax[0].set_title("Standard Deviation, Hell's Half Acre")

ax[1].set_xlabel("x distance")
ax[1].set_ylabel("y diatance")
ax[1].set_title("Average, Hell's Half Acre")

plt.show()

dw.write_band(
    each_gdal,
    average,
    "C:\\Users\\palatyle\\Documents\\Tube-detect",
    "Average.tiff",
    None,
)
