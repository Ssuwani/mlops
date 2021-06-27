from PIL import Image
import numpy as np
img1 = Image.open("/home/suwan/Downloads/dog.png")
img2 = Image.open("/home/suwan/Downloads/profile_image.jpg")

print(np.array(img1).shape)
print(np.array(img2).shape)