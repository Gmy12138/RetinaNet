
import glob
import os

dir = '/home/ecust/gmy/pytorch-retinanet-master/pytorch-retinanet-master/detection_result'  # xml目录

a=sorted(glob.glob("%s/*.*" % dir))
# print(a)
for i in a:
    f = open(i, "r+")
    f.truncate()
