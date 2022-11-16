import face_recognition
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
import os,shutil
from tqdm import tqdm

index = 0
data_root = r'C:\temp_can\5242test\new_data\without_mask'
for root,dirs,files in os.walk(data_root,topdown=False):
    for name in tqdm(files):
        try:
            image = face_recognition.load_image_file(os.path.join(root, name))
            face_locations = face_recognition.face_locations(image)
            # print(face_locations)

            for i in range(len(face_locations)):
                loc_tuple = (face_locations[i][3], face_locations[i][0], face_locations[i][1], face_locations[i][2]) # Use face_recognition library to crop human face. The output is the face location of each face, here is how to crop it.
                img = Image.open(os.path.join(root, name))
                img_new = img.crop(loc_tuple)
                if(img_new.width>50 and img_new.height>50):
                    img_new.save(r'C:\temp_can\delete2\better_face_no_mask\{}.jpg'.format(index))
                    index += 1
        except:
            print(name)






