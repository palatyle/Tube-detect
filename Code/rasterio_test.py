import rasterio
import matplotlib.pyplot as plt
fn = '/Users/tylerpaladino/Documents/ISU/Meteor_Crater_EVA/S2A_MSIL2A_20211011T180301_N0301_R041_T12SVD_20211011T205452.SAFE/GRANULE/L2A_T12SVD_A032928_20211011T180700/IMG_DATA/R10m/T12SVD_20211011T180301_B02_10m.jp2'
dataset = rasterio.open(fn)

band = dataset.read(1)

plt.imshow(band, interpolation='antialiased')
print('test')