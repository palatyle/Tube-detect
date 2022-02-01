from matplotlib.colors import rgb_to_hsv
import rasterio as rio
from rasterio.plot import show, show_hist
import matplotlib.pyplot as plt
import earthpy.plot as ep 
import os
import numpy as np

src = rio.open('/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/Code/data/day_ortho_16bit.tiff')


red = src.read(3, masked=True)
NIR = src.read(5, masked=True)
NDVI = np.zeros(red.shape, dtype=rio.float32)
NDVI = (NIR.astype(float)-red.astype(float))/(NIR+red)

os.chdir('/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/Code')
with rio.Env():
    profile =src.profile
    profile.update(dtype=rio.float32, count=1)
    with rio.open('NDVI.tif', 'w', **profile) as dst:
        dst.write(NDVI.astype(float),1)

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
