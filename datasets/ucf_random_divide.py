import os
from glob import glob
import random
root_path = "../ucf"
im_list = sorted(glob(os.path.join(root_path, '*.jpg')))
test_list = random.choices(im_list,k=10)
for i in test_list:
    #print(i,im_list)
    im_list.remove(i)
    
for i in test_list:
    # os.system("mv "+i.split('.')[0]+" ../ucf/test")
    #print("mv "+i.replace(".jpg", "_ann.mat")+" ../ucf/test")
    os.system("mv "+i+" ../ucf/test")
    os.system("mv "+i.replace(".jpg", "_ann.mat")+" ../ucf/test")