#导入需要用到的模块
import paddle
import paddle.nn as nn
import pandas as pd
import numpy as np
import random
import json
import shutil
import os
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageChops

test_json = '/data2/competion/tian_dianwang/3_testa_user.csv'

df = pd.read_csv(test_json,header=0)
df = df["image_url"]
print(df[0])

path = "/data2/competion/tian_dianwang"
new_path = "/data2/competion/tian_dianwang/3_test_imagesa_result_cascade_paddle"
if not os.path.exists(new_path):
    os.makedirs(new_path)
    
dict_lable = {
    "1":"badge",
    "2":"safebelt",
    "3":"offgroundperson",
  
}

#提交结果保存路径
results_json_path = "/data2/competion/tian_dianwang/PaddleDetection/results_cascade_paddle.json"

json_04 = pd.read_json(results_json_path)
#返回唯一图片名字
json_id = json_04['image_id'].unique()
for ids,one_json_id in enumerate(json_id):
    #取出一张图片的全部框
    num_json = json_04[json_04['image_id']==one_json_id]
    
    img_name = df[one_json_id]
    print(img_name)
    img = cv2.imread(os.path.join(path,img_name),1)
    print('num_json = ')
    print(num_json)    

    if len(num_json) > 1:
        
        for farme_id in range(len(num_json)):
            
            one_farme = num_json.iloc[farme_id]
            #print(one_farme)
            
            category_id = dict_lable[str(one_farme["category_id"])]
            print(ids,one_farme["category_id"], category_id)
            # print(ids,img_name,print(num_json))
            
            bbox = one_farme["bbox"]
            # print(js["category_id"])
            # bbox = np.array(bbox[0])
            # print(bbox)
            # print(os.path.join(path,img_name))
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            if one_farme["category_id"] == 1:
                cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
            elif one_farme["category_id"] == 2:
                cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            elif one_farme["category_id"] == 3:
                cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
            cv2.putText(img,category_id,((int(x1),int(y1))),cv2.FONT_HERSHEY_PLAIN,4,(0,0,255),4)
            
        if ids < 10:
            cv2.imwrite(os.path.join(new_path,str(ids) + " " + img_name.split("/")[-1]),img)
        else:
            break
        
    else:
        one_farme = num_json
        img_name = df[one_json_id]
        print(ids,one_farme["category_id"].iloc[0])
        # print(ids,img_name,print(num_json))
        
        bbox = one_farme["bbox"]
        category_id = dict_lable[str(one_farme["category_id"].iloc[0])]
        print(ids,one_farme["category_id"].iloc[0], category_id)
        # print(js["category_id"])
        # print(bbox.iloc[0])
        bbox = bbox.iloc[0]
        
        # print(os.path.join(path,img_name))
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        if one_farme["category_id"].iloc[0] == 1:
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
        elif one_farme["category_id"].iloc[0] == 2:
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        elif one_farme["category_id"].iloc[0] == 3:
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
        cv2.putText(img,category_id,((int(x1),int(y1))),cv2.FONT_HERSHEY_PLAIN,4,(0,0,255),4)
        # cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
        #cv2.putText(img,"rectangle",(100,100),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),1)
        
        if ids < 10:
            cv2.imwrite(os.path.join(new_path,str(ids) + " " + img_name.split("/")[-1]),img)
        else:
            break