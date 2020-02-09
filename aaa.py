import os
from PIL import Image
import numpy as np
imgs = os.listdir("F:/dataset")
num = len(imgs)
for i in range(num):
    # L_path = 'C:/Users/Administration/Desktop/hhh/b.jpg'
    L_image = Image.open("F:/dataset/"+imgs[i])
    out = L_image.convert("RGB")
    img = np.array(out)
    # print(type(img))
    # np.savetxt("F:/dataset/output.txt",img)
    f = open("F:/dataset/output.txt","w+")
    # with open("F:/dataset/output.txt", "w") as f:
    #     f.write(str(img))
    f.write(imgs[i])
    # f.write(str(img))
    # print(imgs[i])
    # print(img)


