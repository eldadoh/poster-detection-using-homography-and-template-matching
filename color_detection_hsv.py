from image_plots import plot_img_opencv
import cv2
import numpy as np

def hsv_color_detection(img_path,Color,show = False):

    img = cv2.imread(img_path)
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mask1 = None
    
    if Color =='RED':
        
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    
        lower_red = np.array([170,50,50])
        upper_red = np.array([180,255,255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    
    elif Color =='BLUE':

        lower_blue =np.array([100,150,0],np.uint8)
        upper_blue =np.array([140,255,255],np.uint8)
        mask0 = cv2.inRange(img_hsv, lower_blue, upper_blue)

    elif Color == 'WHITE' :

        lower_white = np.array([0,0,0], dtype=np.uint8)
        upper_white = np.array([0,0,255], dtype=np.uint8)
        mask0 = cv2.inRange(img_hsv, lower_white, upper_white)

    if mask1 is not None : 
        mask = mask0 + mask1
    else : 
        mask = mask0

    # set my output img to zero everywhere except my mask
    output_img = img.copy()
    output_img[np.where(mask==0)] = 0

    # or your HSV image, which I *believe* is what you want
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask==0)] = 0   

    if show :
        
        plot_img_opencv(output_img,width=500)


    return output_img 

def main():

    poster_planogram_image = 'Data_new/planograms/planograms_parsed_images/APPBARBSDMN24x150421.jpg'
    poster_red_img  = 'Data_new/planograms/planograms_parsed_images/red/TWSFSAMN24X150421.jpg'
    poster_blue_img = 'Data_new/planograms/planograms_parsed_images/blue/N12E5416.jpg'

    scene_image_O = '/home/arpalus/Work_Eldad/Arpalus_Code/Eldad-Local/arpalus-poster_detection/Data_new/realograms/valid_jpg_format_realograms_images/IMG_1559.jpg'
    scene_image_lot_of_red = 'Data_new/realograms/valid_jpg_format_realograms_images/IMG_1558.jpg'
    
    hsv_color_detection(scene_image_lot_of_red,Color='RED',show = True)

if __name__ == "__main__":

    main()
