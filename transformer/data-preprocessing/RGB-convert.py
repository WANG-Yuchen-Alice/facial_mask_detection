from PIL import Image
import os,shutil

# To convert non-RGB photo into RGB mode. Or delete too small photo


data_root = r'C:\temp_can\delete2\better_face_new_data\better_face_no_mask'
for root,dirs,files in os.walk(data_root,topdown=False):
    # print(len(files))
    for name in files:
        try:
            img = Image.open(os.path.join(root, name))
            if img.mode == 'RGB':
                if img.height+img.width>100:
                    shutil.copy(os.path.join(root, name), r'C:\temp_can\delete2\better_face_new_data\new_rgb_data\no_mask')
            else:
                if img.height+img.width>100:
                    img = img.convert("RGB")
                    img.save(r'C:\temp_can\delete2\better_face_new_data\new_rgb_data\no_mask')
        except:
            print(os.path.join(root, name))
    print(len(files))
