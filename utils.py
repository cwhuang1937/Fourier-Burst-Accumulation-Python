import numpy as np
import cv2
import glob
import os
from matplotlib import pyplot as plt
import math


def read_burst(burst_path, file_extension):
    """
    Read a burst of images with extension <file_extension>
    return a burst of images
    """
    image_dirs = glob.glob(os.path.join(burst_path, file_extension))
    image_dirs = sorted([dir for dir in image_dirs if dir.find('out') == -1])
    return np.array([cv2.imread(name) for name in image_dirs], dtype=np.uint8)


def register_burst(burst, copy=False):
    """
    allign a burst of images
    source: https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
    """
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    if copy:
        burst = np.copy(burst)

    # Convert images to grayscale
    burst_gray = np.array([cv2.cvtColor(burst[i], cv2.COLOR_BGR2GRAY)
                          for i in range(burst.shape[0])])

    # Detect ORB features and compute descriptors.
    sift = cv2.SIFT_create(MAX_FEATURES)
    keypoints = [0 for i in range(burst.shape[0])]
    descriptors = [0 for i in range(burst.shape[0])]
    for i in range(burst.shape[0]):
        keypoints[i], descriptors[i] = sift.detectAndCompute(
            burst_gray[i], None)

    (height, width) = burst[0].shape[:2]

    for i in range(1, burst.shape[0]):

        #         # Match features.
        #         # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        #         matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        #         matches = list(matcher.match(descriptors[0], descriptors[i], None))

        #         # Sort matches by score
        #         matches.sort(key=lambda x: x.distance, reverse=False)

        #         # Remove not so good matches
        #         numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        #         matches = matches[:numGoodMatches]

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors[0], descriptors[i], k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        matches = good

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for j, match in enumerate(matches):
            points1[j, :] = keypoints[0][match.queryIdx].pt
            points2[j, :] = keypoints[i][match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        burst[i] = cv2.warpPerspective(burst[i], h, (width, height))

    return burst


def get_gau_ker(ksize, sig, shape=None):
    # get gaussian blur kernel
    # return (the space domain of the blur kernel, the frequency domain of the blur kernel)
    if shape == None:
        shape = (ksize, ksize)

    # calculate kernel
    l = ksize//2
    c, r = np.meshgrid(np.linspace(-l, l, ksize), np.linspace(-l, l, ksize))
    gauss = np.exp(-(np.square(c)+np.square(r))/2/(sig**2))
    gauss /= gauss.sum()

    # pad kernel
    res = np.zeros(shape, dtype=np.float64)
    rmid = shape[0]//2
    cmid = shape[1]//2
    res[rmid-l:rmid+l+1, cmid-l:cmid+l+1] = gauss
    return res, np.abs(np.fft.fft2(res))


def gaussian_sharpen(Input, radius, c):
    # Input : image
    # radius : radius of H
    # c : images sharpening constant
    img = Input.copy()
    centx = int(Input.shape[0]/2)
    centy = int(Input.shape[1]/2)
    print("max max radius: "+str(math.sqrt(centx**2+centy**2)) +
          " min max radius: "+str(min(centx, centy)))
    img = np.moveaxis(img, 2, 0)  # (R, C, channel) -> (channel, R, C)
    img_result = np.zeros(img.shape)
#     print(img.shape)
    for channel in range(img.shape[0]):
        img_fft = np.fft.fft2(img[channel])
        img_fft = np.fft.fftshift(img_fft)
        for i in range(img_fft.shape[0]):
            for j in range(img_fft.shape[1]):
                cur_rad = (i-centx)**2+(j-centy)**2
                scale = math.exp(-cur_rad / (2*(radius**2)))
                img_fft[i, j] = scale * img_fft[i, j]
        img_result[channel] = np.real(np.fft.ifft2(np.fft.ifftshift(img_fft)))
    img_result = np.moveaxis(img_result, 0, 2)
#     print(img_result.shape)
    img_result = np.where(img_result < 0, 0, img_result)
    img_result = np.where(img_result > 255, 255, img_result).astype(np.uint8)
    # image sharpening
    img_result = (c/(2*c-1))*Input - ((1-c)/(2*c-1))*img_result
    return img_result.astype(np.uint8)


def unsharp_masking(img, c=3/5):
    # img_median = cv2.medianBlur(img, 1)
    # img_lap = cv2.Laplacian(img, cv2.CV_64F)
    # img_sharp = img_median - 0.7*img_lap
    blur_img = cv2.GaussianBlur(img, (5, 5), 0)
    img_result = c/(2*c-1)*img - (1-c)/(2*c-1)*blur_img
    img_result = np.where(img_result < 0, 0, img_result)
    img_result = np.where(img_result > 255, 255, img_result).astype(np.uint8)
    return img_result


def change_reference(burst, copy=False):
    """
    Use the clearest image as the reference image for burst registration
    Put the clearest image first
    burst : a burst of images
    """
    if copy:
        burst = np.copy(burst)
    a = np.fft.fft2(np.moveaxis(burst, 3, 1))
    a = np.sum(np.abs(a), axis=(1, 2, 3))
    a = np.argmax(a)
    burst[[0, a], :, :, :] = burst[[a, 0], :, :, :]
    return burst


def paper_post_processing(img, ksize):
    """
    Use the paper\'s method to do the post-processing.
    img: input image
    ksize: gaussian kernel size
    """
    upb = cv2.fastNlMeansDenoisingColored(
        img, None, 2.5, 2.5, 7, 21).astype(np.float64)
    ups = 2*upb-cv2.GaussianBlur(upb, (ksize, ksize), 3.5)
    upr = ups+0.4*(img-upb)
    upr = np.where(upr < 0, 0, upr)
    return np.where(upr > 255, 255, upr).astype(np.uint8)


def our_post_processing(img, ksize):
    """
    Use our method to do post-processing.
    img: input image
    ksize: gaussian kernel size
    """
    image_restored = cv2.GaussianBlur(img, (ksize, ksize), 1)
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(src=image_restored, ddepth=-1, kernel=kernel)
