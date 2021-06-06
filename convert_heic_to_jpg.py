import whatimage
import pyheif
from PIL import Image
import io
import glob 
import os 

def decodeImage_from_heic_format_to_jpg(bytesIo,output_path):

    format_ = 'heic'
    if format_ in ['heic', 'avif']:
            i = pyheif.read_heif(bytesIo)

            # Extract metadata etc
            for metadata in i.metadata or []:
                if metadata['type']=='Exif':
                    pass

            # Convert to other file format like jpeg
            s = io.BytesIO()
            pi = Image.frombytes(
                mode=i.mode, size=i.size, data=i.data)

            pi.save(output_path, format="jpeg")



def decode_images_dir(images_dir_path,output_dir_path):

    for img_ in glob.glob(images_dir_path + '/*.heic'):

        img_name = os.path.basename(img_).split('.')[0] + '.jpg'
        path_to_save = os.path.join(output_dir_path,img_name)
        decodeImage_from_heic_format_to_jpg(img_ ,path_to_save )

IMAGES_DIR_PATH = 'Data_new/realograms/Images of the store' 
OUTPUT_DIR_PATH = 'Data_new/realograms/jpg_format'

decode_images_dir(IMAGES_DIR_PATH,OUTPUT_DIR_PATH)