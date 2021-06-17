import os
import cv2
import shutil
from PIL import Image

path = '/home/hyshuai/competion/3_test_imagesa_orientation_ok'
path_write = '/home/hyshuai/competion/3_test_imagesa_orientation_rgb'
if not os.path.exists(path_write):
    os.mkdir(path_write)

list_img = os.listdir(path)
for img_file in list_img:
    img_path =  os.path.join(path, img_file)
    dst_path = os.path.join(path_write, img_file)
    #print('--- img_path={} '.format( img_file))
    
    try:
        image = Image.open(img_path)
        #print('--- img_path={} '.format( img_file))
        if image.mode != 'RGB':
            print('--- img_file ={}   image.mode ={} '.format( img_file, image.mode))
            #print('---  dst_path={} '.format( dst_path))            
            print(image.format, image.mode, image.size)
            img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGBA2RGB)
            cv2.imwrite(dst_path, img_rgb)
        else:
            shutil.copyfile(img_path, dst_path)
    except (AttributeError, KeyError, IndexError,IOError):
        print(' copy directly , error image -----------: ',img_file)        
        shutil.copyfile(img_path, dst_path)
        pass        
        
    