import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from PIL import Image
from os.path import join

#total_path = '/home/hyshuai/dataset_person/images/val2014'
#images_path = total_path
#sub_path = '/home/hyshuai/dataset_person/labels/val2014'
#label_path_write = '/home/hyshuai/dataset_person/labels/val2014'
#train_empty_txt = open('val2014_empty.txt','w')

#if not os.path.exists(label_path_write):
    #os.makedirs(label_path_write)

#total_filename = []
#for file_n in os.listdir(total_path):
    #total_filename.append(file_n.split('.')[0])
    
#sub_filename = []
#for file_n in os.listdir(sub_path):
    #sub_filename.append(file_n.split('.')[0])
    

#total_filename = set(total_filename)
#sub_filename =  set(sub_filename)
#print(' total_filename = {}'.format(len(total_filename)))
#print(' sub_filename = {}'.format(len(sub_filename)))
#empty_list = list(total_filename - sub_filename)
#print(' empty_list = {}'.format(len(empty_list)))


#for file_n in empty_list:
    #labelpath = os.path.join(label_path_write , file_n+'.txt')
    #out_file = open(labelpath, 'w')
    #out_file.close()    
    
    #img_path = os.path.join(images_path , file_n+'.jpg')
    #train_empty_txt.write(img_path+"\n")


#train_empty_txt.close()
    






# list_file = open('train_no_xml_shaoba2.txt', 'w')
# pathimg='/home/hyshuai/dataset_detection/image_no_xml_shaoba'
# labelpath =pathimg.replace('image_no_xml_shaoba','labels_no_xml_shaoba')
# print('labelpath  = ',labelpath)
#
# if not os.path.exists(labelpath):
#     os.makedirs(labelpath)
# files = os.listdir(pathimg)
# import random
# random.shuffle(files)
#
# print('annotations len = ',len(files))
#
# for file in files:
#
#     imagefile =os.path.join(pathimg,file)
#     print(imagefile)
#
#     try:
#         image = Image.open(imagefile)
#         if image.mode != 'RGB':
#             print(image.format, image.mode, image.size)
#             continue
#     except IOError:
#         print('error : open image error')
#         continue
#     labelpath = imagefile.replace('image_no_xml_shaoba', 'labels_no_xml_shaoba').replace('jpg', 'txt')
#     out_file = open(labelpath, 'w')
#     out_file.close()
#
#     list_file.write(imagefile.replace('image_no_xml_shaoba', 'JPEGImages')+"\n")
#
#
# list_file.close()


list_file = open('test_power3.txt', 'w')
pathimg='/home/hyshuai/competion/3_test_imagesa_orientation_rgb'
label_write_path ='/data2/competion/tian_dianwang/PaddleDetection/dataset/coco/labels/test_power3'
print('label_write_path  = ',label_write_path)

if not os.path.exists(label_write_path):
    os.makedirs(label_write_path)
files = os.listdir(pathimg)
import random
random.shuffle(files)

print('annotations len = ',len(files))

for file in files:

    imagefile =os.path.join(pathimg,file)
    print(imagefile)

    try:
        image = Image.open(imagefile)
        if image.mode != 'RGB':
            print(image.format, image.mode, image.size)
            continue
    except IOError:
        print('error : open image error')
        continue
    labelpath =os.path.join(label_write_path,str(file.split('.')[0])+'.txt')  
    out_file = open(labelpath, 'w')
    out_file.close()
    # if imagefile.find("frame_1") != -1 :
    #     list_file.write(imagefile.replace('images_no_xml', 'JPEGImages') + "\n")

    list_file.write('/coco/images/test_pow3/'+file+"\n")


list_file.close()
