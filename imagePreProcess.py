import os
import cv2
import glob
import pandas as pd

li=list(pd.read_csv("/home/pirl/Downloads/left2right.csv", encoding = "utf-8")['폴더이름'])
print(li)
rootPath="/home/pirl/Documents/splited_action_data2/left2right"

for needName in li:
    objPath = rootPath+"/left2right_"+str(needName)
    print(objPath)
    for file in os.listdir(objPath):
        src = cv2.imread(objPath+"/"+file, cv2.IMREAD_COLOR)
        height, width, channel = src.shape
        matrix = cv2.getRotationMatrix2D((width/2, height/2), 270, 1)
        dst = cv2.warpAffine(src, matrix, (width, height))

        #print(objPath+'/'+file)
        cv2.imwrite(objPath+"/"+file, dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()