import cv2
import numpy as np
import glob
import re
import os

img_array = []

a = sorted(glob.glob('/Users/danemalenfant/PycharmProjects/agent-time-attention-main/src_minigrid/frames/*.png'), key=os.path.getmtime)

print(a)
a = sorted(a, key=lambda x:float(re.findall("(\d+)",x)[0]))
size = 0
for filename in a:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
    img_array.append(img)
    img_array.append(img)
    img_array.append(img)
    img_array.append(img)

out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()



