from matplotlib.colors import rgb_to_hsv
import rasterio as rio
from rasterio.plot import show, show_hist
import matplotlib.pyplot as plt
import os
import numpy as np
from osgeo import gdal



src = rio.open('C:\Students\Paladino\Tube-detect\data\day_ortho_16bit.tif')
src_DEM = 'C:\Students\Paladino\Tube-detect\data\day_DEM.tif'
dest = 'C:\Students\Paladino\Tube-detect\data'


# print("Read in Red channel")
# red = src.read(3, masked=True)
# print("Read in Green channel")
# green = src.read(2, masked=True)
# print("Read in NIR channel")
# NIR = src.read(5, masked=True)

# print("Calculating NDVI...")
# NDVI = np.zeros(red.shape, dtype=rio.float32)
# NDVI = (NIR.astype(float)-red.astype(float))/(NIR+red)
# print("Done!")

# print("Calculating NDWI...")
# NDWI = np.zeros(red.shape, dtype=rio.float32)
# NDWI = (green.astype(float)-NIR.astype(float))/(green+NIR)
# print("Done!")


print('Calculating slope...')
proc_opts = gdal.DEMProcessingOptions(slopeFormat="degree")
gdal.DEMProcessing(dest+"\slope.tif",src_DEM,"slope", options=proc_opts)


os.chdir(dest)

print('Writing data...')
with rio.Env():
    profile = src.profile
    profile.update(dtype=rio.float32, count=1)
    with rio.open('NDVI.tif', 'w', **profile) as dst:
        dst.write(NDVI.astype(float),1)
    with rio.open('NDWI.tif', 'w', **profile) as dst:
        dst.write(NDWI.astype(float),1)

# rgb_stack = np.dstack((red,green,blue))
# plt.imshow(rgb_stack,interpolation='none')
# plt.imshow(src.read(),interpolation='none')
# show(src.read())
# for band in range(1, src.count+1):
#     single_band = src.read(band)

#     # get the output name
#     out_name = os.path.basename("day_ortho")
#     file, ext = os.path.splitext(out_name)
#     name = file + "_" + "B" + str(band) + ".tif"
#     out_img = name

#     print(out_img + " done")

#     # Copy the metadata
#     out_meta = src.meta.copy()

#     out_meta.update({"count": 1})

#     os.chdir('/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/Code')
#     # save the clipped raster to disk
#     with rio.open(out_img, "w", **out_meta) as dest:
#         dest.write(single_band,1)

# b1 = src.read(1)

# with rio.open('b1.tif', "w", **out_meta):

# show(rst)
# plt.imshow(rst.read(),interpolation='none')
# ep.plot_bands(rst,cols=2)
# plt.show()
