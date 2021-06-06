import whatimage
import pyheif
from PIL import Image
import io
import glob 
import os 

def decode_img_from_heic_format_to_jpg(bytesIo,output_path):

    format_ = 'heic'

    if format_ in ['heic', 'avif']:

            i = pyheif.read_heif(bytesIo)

            for metadata in i.metadata or []: # Extract metadata 
                if metadata['type']=='Exif':
                    pass

            pi = Image.frombytes(mode=i.mode, size=i.size, data=i.data)

            pi.save(output_path, format="jpeg")

def decode_heic_to_jpg_for_images_dir(images_dir_path,output_dir_path):

    for img_ in glob.glob(images_dir_path + '/*.heic'):

        img_name = os.path.basename(img_).split('.')[0] + '.jpg'
        path_to_save = os.path.join(output_dir_path,img_name)
        decode_img_from_heic_format_to_jpg(img_ ,path_to_save )

    print('Done converting all the images from heic to jpg')
    
