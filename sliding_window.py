from img_utils import plot_img_matplotlib
import numpy as numpy
from opencv_utils import Calc_hist_grayscale
from typing import List

def crop_via_sliding_window(img = None, Params_crop: tuple = (3,3), show = False):
    """ 
        Input : Params_crop(h,w) (tuple)

        Note : when the img_w,img_h , doesnt divisible by window_h,window_w 
               there is loss of image information of that cols/rows

        use example : crop_via_sliding_window(img ,Params_crop =  (5,5))
    """
    try :
        grey_levels = 256
        img = numpy.random.randint(0,grey_levels, size=(11,11))
    except Exception : 
        pass 

    WINDOW_SIZE_R ,WINDOW_SIZE_C = Params_crop 

    windows_list = []

    for r in range(0,img.shape[0] - WINDOW_SIZE_R, WINDOW_SIZE_R): 
        
        for c in range(0,img.shape[1] - WINDOW_SIZE_C, WINDOW_SIZE_C):
            
            window = img[r:r+WINDOW_SIZE_R,c:c+WINDOW_SIZE_C]
            
            windows_list.append(window)

            if show : 
                print(window)

    return windows_list

def main(): 

    _ = crop_via_sliding_window(Params_crop=[5,5])

    for img in _ :
         
       plot_img_matplotlib(img)
       Calc_hist_grayscale(img,show = True)

    """ 

    for window in windows_list:
        print (window.shape)
        
            (5, 5)
            (5, 5)
            (5, 5)
            (5, 5)

            r=0,c=0
    [[ 63 173 131 205 239]
    [106  37 156  48  81]
    [ 85  85 119  60 228]
    [236  79 247   1 206]
    [ 97  50 117  96 206]]

    r=0,c=5
    [[108 241 155 214 183]
    [202   2 236 183 225]
    [214 141   1 185 115]
    [  4 234 249  95  67]
    [232 217 116 211  24]]

    r=5,c=0
    [[179 155  41  47 190]
    [159  69 211  41  92]
    [ 64 184 187 104 245]
    [190 199  71 228 166]
    [117  56  92   5 186]]

    r=5,c=5
    [[ 68   6  69  63 242]
    [213 133 139  59  44]
    [236  69 148 196 215]
    [ 41 228 198 115 107]
    [109 236 191  48  53]]

    [[ 63 173 131 205 239 108 241 155 214 183  42]
    [106  37 156  48  81 202   2 236 183 225   4]
    [ 85  85 119  60 228 214 141   1 185 115  80]
    [236  79 247   1 206   4 234 249  95  67 203]
    [ 97  50 117  96 206 232 217 116 211  24 242]
    [179 155  41  47 190  68   6  69  63 242 162]
    [159  69 211  41  92 213 133 139  59  44 196]
    [ 64 184 187 104 245 236  69 148 196 215  91]
    [190 199  71 228 166  41 228 198 115 107  82]
    [117  56  92   5 186 109 236 191  48  53  65]
    [177 170 114 163 101  54  80  25 112  35  85]]
"""

if __name__ == "__main__":

    main()
    