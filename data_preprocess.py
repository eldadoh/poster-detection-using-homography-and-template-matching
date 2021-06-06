import os 
import glob
import shutil 
from convert_heic_to_jpg import decode_heic_to_jpg_for_images_dir

def create_dir_with_override(dir_path):
    try : 
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    except Exception as e : 
        print(e)
        print('Could not create the desired dir with the corersponding dir path : \n' + f'{dir_path}')

def extract_images_from_dirs(planogram_images_dir_path , defualt_output = 'Data_new/planograms/planograms_parsed_images'):

    create_dir_with_override(defualt_output)
    
    for dir_ in sorted(os.listdir(planogram_images_dir_path)):

        child_dir_path = os.path.join(planogram_images_dir_path,dir_)

        for img_ in sorted(glob.glob(child_dir_path + '/*.png')):
            
            img_name = os.path.basename(img_)[:-len('.png')] + '.jpg'
            img_new_path = os.path.join(defualt_output,img_name)
            shutil.copy(img_,img_new_path)

    print('Done parsing')
    
def main():
    
    """
        create Data_new dir: 
        1.create sub dir planogrmas
        2.create sub dir realograms 
    """
    
    HEIC_IMAGES_DIR_PATH = 'Data_new/realograms/Images of the store' 
    HEIC_OUTPUT_DIR_PATH = 'Data_new/realograms/valid_jpg_format_realograms_images'
    PLANOGRAM_IMAGES_DIR_PATH = 'Data_new/planograms/Garden State Plaza'

    create_dir_with_override(HEIC_OUTPUT_DIR_PATH)
    decode_heic_to_jpg_for_images_dir(HEIC_IMAGES_DIR_PATH,HEIC_OUTPUT_DIR_PATH)
    extract_images_from_dirs(PLANOGRAM_IMAGES_DIR_PATH)

if __name__ == "__main__" : 
    main()