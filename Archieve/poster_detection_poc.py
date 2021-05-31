import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from glob import glob
import os
import shutil
import time
import logging

reuse_model = {}
sift = cv.SIFT_create()

def fill_reuse_model(planogram_image):

    img1 = cv.imread(planogram_image, cv.IMREAD_COLOR)

    planogram_image_name = os.path.splitext(os.path.basename(planogram_image))[0]
    if (planogram_image_name not in reuse_model.keys()):
        # queryImage
        kp1, des1 = sift.detectAndCompute(img1,None)
        reuse_model[planogram_image_name] = [kp1, des1]


def detect(planogram_image, realogram_image, show=False):
    MIN_MATCH_COUNT = 10
    MAX_MATCH_DIST = 100
    MIN_MATCH_NUM = 30
    MIN_DETECT_AREA = 50
    planogram_image_name = os.path.splitext(os.path.basename(planogram_image))[0]

    img1 = cv.imread(planogram_image, cv.IMREAD_COLOR)
    img2 = cv.imread(realogram_image, cv.IMREAD_COLOR) # trainImage

    if (planogram_image_name in reuse_model.keys()):
        kp1, des1 = reuse_model[planogram_image_name]
    else:
         # queryImage
        print("Create model - should not reach here if fill_reuse_model called")
        kp1, des1 = sift.detectAndCompute(img1,None)
        reuse_model[planogram_image_name] = [kp1, des1]


    #img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    #img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

    # Initiate SIFT detector

    # find the keypoints and descriptors with SIFT


    h,w,d = img2.shape
    logging.info('1 image size h-' + str(h) + " w-" + str(w))
    kp2, des2 = sift.detectAndCompute(img2,None)
    logging.info('2')
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    logging.info('3')
    matches = flann.knnMatch(des1,des2,k=2)
    logging.info('4')
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    logging.info('5')
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        logging.info('6')
        if M is not None:
            matchesMask = mask.ravel().tolist()
            h,w,d = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts, M)
            logging.info('7')
            d_w = dst[3,0,0] - dst[0,0,0]
            d_h = dst[1,0,1] - dst[0,0,1]
            d_area = d_w*d_h



            dst_pts_in = dst_pts[(mask==1).ravel(), :, :]
            src_pts_in = src_pts[(mask==1).ravel(), :, :]
            match_quality = np.linalg.norm(dst_pts_in - cv.perspectiveTransform(src_pts_in, M))
            match_num = np.sum(mask)

            logging.info('8')
            img_with_detection = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
            cv.putText(img_with_detection, planogram_image_name  + " " + str(int(match_quality)) + " " + str(int(match_num)),  (0, 0 + 45), cv.FONT_HERSHEY_SIMPLEX, 2, (36,255,12), 2) #(int(dst[0,0,0]), int(dst[0,0,1])

            logging.info('9')
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                               singlePointColor = None,
                               matchesMask = matchesMask, # draw only inliers
                               flags = 2)

            img_with_matches = cv.drawMatches(img1,kp1,img_with_detection,kp2,good,None,**draw_params)

            logging.info('10')
            if show:
                plt.imshow(img_with_matches, 'gray'),plt.show()

            if match_quality < MAX_MATCH_DIST and match_num > MIN_MATCH_NUM and d_area > MIN_DETECT_AREA:
                return True, match_quality, match_num, d_area, img_with_detection, img_with_matches
            else:
                return False, match_quality, match_num, d_area, img_with_detection, img_with_matches
        else:
            print("findHomography failed")


    else:
        print("Not enough matches found in the first phase - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    return False, 0, 0, 0, None, None

def detect_all(planogram_images, realogram_images, output_folder, show=False):

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)

    pos_output_folder = output_folder + "/positive/"
    neg_output_folder = output_folder + "/negative/"
    os.mkdir(pos_output_folder)
    os.mkdir(neg_output_folder)

    start_time = time.time()
    for planogram_image in planogram_images:
        fill_reuse_model(planogram_image)
    print("--- Done create models in %s seconds ---" % (time.time() - start_time))


    for realogram_image in realogram_images:
        found_match = False
        for planogram_image in planogram_images:
            planogram_image_name = os.path.splitext(os.path.basename(planogram_image))[0]
            realogram_image_name = os.path.splitext(os.path.basename(realogram_image))[0]
            start_time = time.time()
            logging.info("******* start detection *******")
            is_match, match_quality, match_num, d_area, img_with_detection, img_with_matches = detect(planogram_image, realogram_image, show)
            logging.info("--- detection took %s seconds ---" % (time.time() - start_time))
            logging.info("planogram_image_name = {} realogram_image_name = {} is_match = {} match_quality = {} match_num = {} d_area = {}".format(planogram_image_name, realogram_image_name, is_match, match_quality, match_num, d_area))

            if is_match:
                cv.imwrite(output_folder + realogram_image_name + "_" + planogram_image_name + "_detection.jpg", img_with_detection)
                cv.imwrite(output_folder + realogram_image_name + "_" + planogram_image_name + "_matches.jpg", img_with_matches)
                found_match = True

        orig_img = cv.imread(realogram_image, cv.IMREAD_COLOR)
        if found_match:
            cv.imwrite(pos_output_folder + realogram_image_name + ".jpg", orig_img)
        else:
            cv.imwrite(neg_output_folder + realogram_image_name + ".jpg", orig_img)




if __name__ == "__main__":
    logging.basicConfig( format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    planogram_images = glob("/media/dov84d/EHD3/ARPalus/Verizon/Posters/planogram_images/*.png")
    realogram_images = glob("/media/dov84d/EHD3/ARPalus/Verizon/Posters/realogram_images/*.jpg")
    output_folder = "/media/dov84d/EHD3/ARPalus/Verizon/Posters/results/"
    detect_all(planogram_images, realogram_images, output_folder)
    #detect(show=True)

