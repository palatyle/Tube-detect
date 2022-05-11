# %matplotlib inline             I commented this out because the internet said I could replace it with lines 3-4.
import matplotlib.pyplot as plt
new_obj.resample('M').sum().plot(kind="bar")
plt.show()
import numpy as np
import os 
import PIL
from PIL import Image

for i in range(1, 21):
    col = int(255 * i / 20)
    img = Image.new('RGB', (120, 80), color=(col, col, col))
    img.save('%02d.bmp' % i, format='bmp')

ims = []
for i in range(1, 21):
    ims.append(Image.open('%02d.bmp' % i, mode='r'))

ims[0] # 1, -1  out

ims = np.array([np.array(im) for im in ims])
ims.shape #out
imave = np.average(ims,axis=0)
imave.shape #out
result = Image.fromarray(imave.astype('uint8'))
result.save('result.bmp')