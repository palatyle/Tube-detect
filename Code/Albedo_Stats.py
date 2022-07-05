import matplotlib.pyplot as plt
import numpy as np
import os
import data_wrangle as dw

'''
first_file = dw.GDAL_read_tiff("E:\Data\HHA_Calculated_Albedo\AlbedoS2A_MSIL2A_20190511T181921_N0212_R127_T12TUP_20190511T224452_super_resolved.tif")
first_raster = dw.GDAL2NP(first_file)

second_file = dw.GDAL_read_tiff("E:\Data\HHA_Calculated_Albedo\AlbedoS2A_MSIL2A_20190720T181931_N0213_R127_T12TUP_20190721T001757_super_resolved.tif")
second_raster = dw.GDAL2NP(second_file)

print("done")

test = np.stack([first_raster, second_raster])

test_average = np.average(test, axis=0)

test_std = np.std(test, axis = 0)
print("nice")
'''




working_directory = "E:\Data\HHA_Calculated_Albedo"
os.chdir(working_directory)
working_list = os.listdir(working_directory)
array_list = []

for eachfile in working_list:
    each_gdal = dw.GDAL_read_tiff(eachfile)
    each_raster = dw.GDAL2NP(each_gdal)

    # listtoarray_raster = np.array(each_raster)
    array_list.append(each_raster)

    print("rasters initalized")

    print("done!")


full_stack = np.stack(array_list, axis = 0) 
average = np.average(full_stack, axis = 0)
std_dev = np.std(full_stack, axis = 0)
print("Done!")

fig, ax = plt.subplots()
img_plot = ax.imshow(std_dev)
fig.colorbar(img_plot)
ax.set_xlabel("x distance")
ax.set_ylabel("y diatance")
ax.set_title("Standard Deviation, Hell's Half Acre")

plt.show()

dw.write_band(each_gdal, average, "E:\Data\Georeference_Outputs", "Average.tiff" )
