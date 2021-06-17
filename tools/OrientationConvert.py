from PIL import Image, ExifTags
import os
import shutil

path_img ='/data2/competion/tian_dianwang/3_test_imagesa'
path_write ='/home/hyshuai/competion/3_test_imagesa_orientation_ok'
if not os.path.exists(path_write):
    os.mkdir(path_write)
files = os.listdir(path_img)
for file_n in files:
    filepath = os.path.join(path_img,file_n)
    filepath_write = os.path.join(path_write,file_n)
    if os.path.isfile(filepath_write):
        continue

    try:
        image=Image.open(filepath)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(image._getexif().items())
    
        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
            print(file_n)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
            print(file_n)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
            print(file_n)
        image.save(filepath_write)
        image.close()
    
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        print(' copy directly , error image dont have getexif : ',file_n)        
        shutil.copyfile(filepath, filepath_write)        
        pass