import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
import os
import time
import logging
from opencv_utils import drawKeyPts_single_image
from image_plots import plot_img_opencv
import numpy as np
from template_matching import calc_ssim, template_matching_func
from skimage.transform import rescale, downscale_local_mean
from skimage.transform import resize as skimage_resize 




model = {}
# nfeatures=None, nOctaveLayers=None, contrastThreshold=None, edgeThreshold=None, sigma=None)
detector_descriptor = cv.SIFT_create()

# detector_descriptor = cv.ORB_create()

base_path_for_gt = "Data/annotations/"
model_scale_percent = 100
input_height = 800
iou_th = 0.3


# box (x_left, y_top, x_right, y_bottom)
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def resize_input_img(img):
    height = img.shape[0]
    width = int(img.shape[1] * input_height / height)
    height = input_height
    dsize = (width, height)
    img = cv.resize(img, dsize)
    return img


def get_gt_for_img(realogram_image, filter_classes=[0], show=False):
    realogram_image_name = os.path.splitext(
        os.path.basename(realogram_image))[0]
    try:
        annotations = np.loadtxt(
            base_path_for_gt + realogram_image_name + ".txt")
    except:
        print("No annotation for ", realogram_image)
        return

    img = cv.imread(realogram_image)
    img = resize_input_img(img)

    height = img.shape[0]
    width = img.shape[1]
    result = {}
    if len(annotations.shape) == 1:
        annotations = np.expand_dims(annotations, axis=0)
    for annotation_idx in range(annotations.shape[0]):
        annotation = annotations[annotation_idx, :]

        class_ = annotation[0]
        if str(int(class_)) in filter_classes:
            a_x1 = annotation[1]
            a_y1 = annotation[2]
            a_width = annotation[3]
            a_height = annotation[4]

            x1 = (a_x1 - a_width/2)*width
            y1 = (a_y1 - a_height/2)*height
            x2 = x1 + a_width*width
            y2 = y1 + a_height*height

            result[str(int(class_))] = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]

        if show:
            cv.rectangle(img, (int(x1), int(y1)),
                         (int(x2), int(y2)), (0, 0, 255), 7)
            cv.putText(img, str(int(class_)), (int(x1), int(y1) + 45),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (36, 255, 12), 10)

    if show:
        plot_img_opencv(img)

    return result


def load_model(planogram_images):
    
    print('----------------Start building model dict--------------------')

    for planogram_image in planogram_images:

        img1 = cv.imread(planogram_image, cv.IMREAD_COLOR)
        
        img1 = pad_image_on_borders(img1)
    
        width = int(img1.shape[1] * model_scale_percent / 100)
        height = int(img1.shape[0] * model_scale_percent / 100)
        dsize = (width, height)
        img1 = cv.resize(img1, dsize)
        h, w, d = img1.shape

        planogram_image_name = os.path.splitext(os.path.basename(planogram_image))[0]
        kp1, des1 = detector_descriptor.detectAndCompute(img1, None)

        ###Plot###
        poster_with_kp = drawKeyPts_single_image(img1.copy(), kp1, (0, 255, 0), 5)
        path_to_save_img = os.path.join('posters_keypoints_sift', os.path.basename(planogram_image))
        # cv.imwrite(path_to_save_img, poster_with_kp)
        ##########

        # plot_img_opencv(cv.cvtColor(poster_with_kp,cv.COLOR_BGR2RGB))
        model[planogram_image_name] = [kp1, des1, img1, h, w, d]

    print('-------------Done building model dict------------------')

def detect(realogram_image, show=False):
    
    print('start detection for scene image : ' +f'{realogram_image}')
    
    MIN_MATCH_COUNT = 4
    MAX_MATCH_DIST = 100
    MIN_MATCH_NUM = 4
    MIN_DETECT_AREA = 10

    # gt_for_img = get_gt_for_img(realogram_image, filter_classes=list(model.keys()), show=False)
    realogram_image_name = os.path.splitext(os.path.basename(realogram_image))[0]

    # if gt_for_img is None:
    #     return False, 0,0,0

    img2 = cv.imread(realogram_image, cv.IMREAD_COLOR)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    img2 = resize_input_img(img2)

    # find the keypoints and descriptors with SIFT

    #h,w,d = img2.shape
    kp2, des2 = detector_descriptor.detectAndCompute(img2, None)

    ####Plot#####
    poster_with_kp = drawKeyPts_single_image(img2.copy(), kp2, (0, 255, 0), 5)
    # plot_img_opencv(cv.cvtColor(poster_with_kp,cv.COLOR_BGR2RGB))
    #############0
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    FP, FN, TP = 0, 0, 0
    for planogram_image_name in model.keys():

        kp1, des1, img1, h, w, d = model[planogram_image_name]

        matches = flann.knnMatch(np.float32(des1), np.float32(des2), k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:

            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            if M is not None:

                matchesMask = mask.ravel().tolist()

                pts = np.float32(
                    [[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                # (x_left, y_top), (x_left, y_bottom), (x_right, y_bottom), (x_right, y_top)
                dst = cv.perspectiveTransform(pts, M)

                d_w = dst[3, 0, 0] - dst[0, 0, 0]
                d_h = dst[1, 0, 1] - dst[0, 0, 1]
                d_area = d_w*d_h

                dst_pts_in = dst_pts[(mask == 1).ravel(), :, :]
                src_pts_in = src_pts[(mask == 1).ravel(), :, :]
                match_quality = np.linalg.norm(dst_pts_in - cv.perspectiveTransform(src_pts_in, M))
                match_num = np.sum(mask)

                # iou = 0
                # if planogram_image_name in gt_for_img.keys():
                #     cur_gt_for_img = gt_for_img[planogram_image_name]
                #     if show:
                #         img2 = cv.polylines(img2,[np.int32(cur_gt_for_img)],True,(0,255,0),3, cv.LINE_AA)
                #     box_det = (dst[0,0,0], dst[0,0,1], dst[2,0,0], dst[2,0,1])
                #     box_gt = (cur_gt_for_img[0][0], cur_gt_for_img[0][1], cur_gt_for_img[2][0], cur_gt_for_img[2][1])
                #     iou = bb_intersection_over_union(box_det, box_gt)

                bbox_area = h*w
                print("Results for planogram_image_name = {} realogram_image_name = {} match_quality = {} match_num = {} d_area = {}".format(planogram_image_name, realogram_image_name, match_quality, match_num, d_area))
                print('BBOX AREA IS: ' +f'{bbox_area}')
                if match_quality < MAX_MATCH_DIST and match_num > MIN_MATCH_NUM and d_area > MIN_DETECT_AREA:
                    
                    if show:
                    
                        img_with_detection = cv.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)
                        # cv.putText(img_with_detection, planogram_image_name  + " " + str(int(match_quality)) + " " + str(int(match_num)) + " iou " + str(iou),  (0, 0 + 45), cv.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)

                        
                        # cv.putText(img_with_detection, planogram_image_name + " " + str(int(match_quality)) + " " + str(int(match_num)),  (0, 0 + 45), cv.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)
                        # plt.imshow(img_with_detection, 'gray'), plt.show() 
                        
                        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                           singlePointColor=None,
                                           matchesMask=matchesMask,  # draw only inliers
                                           flags=2)
                        
                        img_with_matches = cv.drawMatches(img1, kp1, img_with_detection, kp2, good, None, **draw_params)
                        
                        plot_img_opencv(cv.cvtColor(img_with_matches,cv.COLOR_BGR2RGB))
                        
                        #plt.imshow(img_with_matches, 'gray'), plt.show()
                    # if iou > iou_th: #TP
                    #     TP += 1
                    #     gt_for_img.pop(planogram_image_name, None)
                    # else:
                    #     FP += 1

            else:
                print("findHomography failed")

        else:
            print(
                "Not enough matches found in the first phase - {}/{}".format(len(good), MIN_MATCH_COUNT))

    # FN = len(gt_for_img.keys())
    return True, FP, FN, TP


def detect_all(planogram_images, realogram_images, show=False):

    start_time = time.time()

    load_model(planogram_images)

    # check number of kp detected on resized images     
    
    # for key_ in model.keys():
    #     num_of_kp_for_current_poster_image = len(model[key_][0])
    #     print('number of kp detected for' +f'{key_}' 'is: ' +f'{num_of_kp_for_current_poster_image}' )

    print("--- Done create models in %s seconds ---" % (time.time() - start_time))

    total_detection_time = 0
    all_FP, all_FN, all_TP = 0, 0, 0
    
    for realogram_image in realogram_images:

        start_time = time.time()
       
        print("******* start detection *******")

        success, FP, FN, TP = detect(realogram_image, show)
      
        if success:
            detection_time = time.time() - start_time
            total_detection_time += detection_time
            print("--- detection took %s seconds ---" % (detection_time))
            all_FP += FP
            all_FN += FN
            all_TP += TP
        else:
            print("Failed - might me missing annotation")

    # print("Average detection time {}".format(
    #     total_detection_time/len(realogram_images)))
    # print("Found TP {} FP {} FN {}".format(
    #     str(all_TP), str(all_FP), str(all_FN)))

def main(): 
    
    ############################################################################################

    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    #exp1 : all 

    # planogram_images = glob("Data_new/planograms/planograms_parsed_images/*.jpg")
    # realogram_images = glob("Data_new/realograms/valid_jpg_format_realograms_images/*.jpg")


    # realogram_image ='/home/arpalus/Work_Eldad/Arpalus_Code/Eldad-Local/arpalus-poster_detection/Data_new/realograms/valid_jpg_format_realograms_images/IMG_1559.jpg'
    # realogram_images = [realogram_image]

    # planogram_image_O_easy = 'Data_new/planograms/planograms_parsed_images/APPBARBSEMN39x270421.jpg'
    # planogram_image_U_hard = 'Data_new/planograms/planograms_parsed_images/APPBARBSDMN24x150421.jpg'
    # planogram_images = [planogram_image_O_easy]

    # detect_all(planogram_images, realogram_images, show=True)
    

    ############################################################################################


    #exp2 : ssim 
    # poster_planogram_image = 'Data_new/planograms/planograms_parsed_images/APPBARBSDMN24x150421.jpg'
    # scene_image = '/home/arpalus/Work_Eldad/Arpalus_Code/Eldad-Local/arpalus-poster_detection/Data_new/realograms/valid_jpg_format_realograms_images/IMG_1559.jpg'
    # template_matching_func(scene_image,poster_planogram_image,output_path='Output/template_matching_output',show = True)
    # calc_ssim(poster_planogram_image,scene_image,show = True)

    ############################################################################################
    
    #exp3: resolution of poster images at different scales 

    posters_planograms_at_different_scales_dir_path = glob.glob('Resulotion_test_data/resized_images/*.jpg')
    
    scene_image1 = 'Resulotion_test_data/scene_test_images/IMG_1547.jpg'
    scene_image2 = 'Resulotion_test_data/scene_test_images/IMG_1545.jpg'
    scene_image3 = 'Resulotion_test_data/scene_test_images/IMG_1546.jpg'
    scene_images = [scene_image1,scene_image2,scene_image3]


    detect_all(posters_planograms_at_different_scales_dir_path, scene_images, show=True)

    ############################################################################################

if __name__ == "__main__":

    main()