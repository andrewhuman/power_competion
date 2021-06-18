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
#求勋章的iou
def badge_iou(preson_frame, thing_frame,p = 0.9):
    x_min = max(preson_frame[0],thing_frame[0])
    y_min = max(preson_frame[1],thing_frame[1])
    x_max = min(preson_frame[2],thing_frame[2])
    y_max = min(preson_frame[3],thing_frame[3])
    # print(x_max-x_min,y_max-y_min)
    # 负值直接为0
    if (x_max-x_min) <= 0 or (y_max-y_min) <= 0:
        return False

    intersection = (x_max-x_min) * (y_max-y_min)
    thing = (thing_frame[2] - thing_frame[0]) * (thing_frame[3] - thing_frame[1])
    iou = intersection/thing
    # print(iou)
    if iou >= p:
        return True
    else:
        return False
#求安全带的iou
def safebelt_iou(preson_frame, thing_frame,p = 0.3):
    x_min = max(preson_frame[0],thing_frame[0])
    y_min = max(preson_frame[1],thing_frame[1])
    x_max = min(preson_frame[2],thing_frame[2])
    y_max = min(preson_frame[3],thing_frame[3])
    # print(x_max-x_min,y_max-y_min)
    # 负值直接为0
    if (x_max-x_min) <= 0 or (y_max-y_min) <= 0:
        return False

    intersection = (x_max-x_min) * (y_max-y_min)
    thing = (thing_frame[2] - thing_frame[0]) * (thing_frame[3] - thing_frame[1])
    iou = intersection/thing
    # print(iou)
    if iou >= p:
        return True
    else:
        return False

#训练集对应类别标注
train_lable = {
    "badge":1,
    "offground":2,
    "ground":3,
    "safebelt":4
}

#测试结果，txt保存路径
path = "/data2/competion/tian_dianwang/PaddleDetection/testA_results"
#测试集图片名字、顺序。官方3_testa_user.csv
test_json = '/data2/competion/tian_dianwang/3_testa_user.csv'
#测试集预测结果保存路径
#test_path = '/data2/competion/tian_dianwang/PaddleDetection/output'
#提交结果保存路径
results_json_path = "results_v5_1.json"

df = pd.read_csv(test_json,header=0)
df = df["image_url"]
# print(df)

results = []#提交结果

for id_s,one_img_name in enumerate(df):
    one_img_name = one_img_name.split("/")[-1].split(".")[0] + ".txt"
    # one_img_path = "/home/aistudio/PaddleDetection/e26c8adc_cb88_4af2_918a_1a0275d39ced.txt"
    one_img_path = os.path.join(path,one_img_name)
    print('\n\n\n')
    print(one_img_name)

    try:#防止空文件,空文件情况直接跳过
        one_txt = pd.read_csv(one_img_path,header=None)
        #one_txt = pd.read_csv("/home/aistudio/PaddleDetection/output/7134a034_0d62_4a2c_a81a_21ff3cdde9ee.txt",header=None)
        
    except:
        print('warning: empty =',one_img_path)
        continue
    one_txt = one_txt[0]
    print('one_txt = {}'.format(one_txt))
    #将2、3两类取出来，即先判断是否是人（天上的加上地上的）
    offground, ground = [], []#人
    badge, safebelt = [], []#物体
    for one_res in one_txt:
        one_res = one_res.split(" ")
        #框的格式化为点的格式x_min,y_min,x_max,y_max
        if int(one_res[0]) == 2:
            frame = np.array([one_res[1], one_res[2],one_res[3],one_res[4],one_res[5]]).astype(np.float32)
            offground.append(np.array([int(frame[1]),int(frame[2]),int(frame[1])+int(frame[3]),int(frame[2])+int(frame[4]),frame[0]]))
        elif int(one_res[0]) == 3:
            frame = np.array([one_res[1], one_res[2],one_res[3],one_res[4],one_res[5]]).astype(np.float32)
            ground.append(np.array([int(frame[1]),int(frame[2]),int(frame[1])+int(frame[3]),int(frame[2])+int(frame[4]),frame[0]]))
        elif int(one_res[0]) == 1:
            frame = np.array([one_res[1], one_res[2],one_res[3],one_res[4],one_res[5]]).astype(np.float32)
            badge.append(np.array([int(frame[1]),int(frame[2]),int(frame[1])+int(frame[3]),int(frame[2])+int(frame[4]),frame[0]]))
        elif int(one_res[0]) == 4:
            frame = np.array([one_res[1], one_res[2],one_res[3],one_res[4],one_res[5]]).astype(np.float32)
            safebelt.append(np.array([int(frame[1]),int(frame[2]),int(frame[1])+int(frame[3]),int(frame[2])+int(frame[4]),frame[0]]))
        else:
            break

    #print(offground)
    #print('ground = {}'.format(ground))
    print('offground = {}'.format(offground))
    print('badge = {}'.format(badge))
    print('safebelt = {}'.format(safebelt))
    
    #判断是否为天上的人
    if len(offground) != 0:
        for off in offground:
            offgroundperson = True#表示为离地的人,也就是第三类,不是作业人员也不是监督人员
            print('len(offground) != 0 offgroundperson =True')
            #判断是否有勋章
            if len(badge) != 0:
                for bad in badge:
                    my_iou = badge_iou(off[0:4],bad[0:4])
                    # print(my_iou)
                    offgroundperson = 1 - my_iou
                    print('badge : offgroundperson ={} my_iou ={} '.format(offgroundperson,my_iou))
                    if my_iou:
                        result = {}
                        result["image_id"] = id_s
                        result["category_id"] = 1
                        result["bbox"] = [off[0],off[1],off[2],off[3]]
                        result["score"] = float(off[4])
                        print('badge = {}'.format(result))
                        results.append(result)

            #判断是否有穿安全带
            if len(safebelt) != 0:
                for safe in safebelt:
                    my_iou = safebelt_iou(off[0:4],safe[0:4])
                    # print(my_iou)
                    offgroundperson = 1 - my_iou
                    print('safebelt : offgroundperson ={} my_iou ={} '.format(offgroundperson,my_iou))
                    if my_iou:
                        result = {}
                        result["image_id"] = id_s
                        result["category_id"] = 2
                        result["bbox"] = [off[0],off[1],off[2],off[3]]
                        result["score"] = float(off[4])
                        print('safebelt = {}'.format(result))
                        results.append(result)
            #
            print('after : offgroundperson ={}  '.format(offgroundperson))     
            result = {}
            result["image_id"] = id_s
            result["category_id"] = 3
            result["bbox"] = [off[0],off[1],off[2],off[3]]
            result["score"] = float(off[4])
            print('offgroundperson = {}'.format(result))
            results.append(result)            
            #if offgroundperson:
                
    #print('results = {}'.format(results))
    #判断是否为地上的人
    #路人是不提交结果的
    print('ground = {}'.format(ground))
    if len(ground) != 0:
        for gro in ground:

            #判断是否有勋章
            if len(badge) != 0:
                for bad in badge:
                    my_iou = badge_iou(gro[0:4],bad[0:4])
                    #print(my_iou)
                    if my_iou:
                        result = {}
                        result["image_id"] = id_s
                        result["category_id"] = 1
                        result["bbox"] = [gro[0],gro[1],gro[2],gro[3]]
                        result["score"] = float(gro[4])
                        results.append(result)
            #判断是否有穿安全带
            if len(safebelt) != 0:
                for safe in safebelt:
                    my_iou = safebelt_iou(gro[0:4],safe[0:4])
                    # print(my_iou)
                    if my_iou:
                        result = {}
                        result["image_id"] = id_s
                        result["category_id"] = 2
                        result["bbox"] = [gro[0],gro[1],gro[2],gro[3]]
                        result["score"] = float(gro[4])
                        results.append(result)

print(len(results))
json.dump(results,open(results_json_path,'w'),indent=4) 