import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def openImage(filename):
    image = cv.imread(filename)
    try:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    except:
        print(" --(!) Error reading image ", filename)
        return None
    return image

def filterMatchesByDistance(matches):
    filteredMatches = []
    matches = sorted(matches, key = lambda x:x.distance)
    ptsPairs = min(400, len(matches)*0.3)
    filteredMatches = matches[:ptsPairs]
    return filteredMatches

def filterMatchesRANSAC(matches, keypointsA, keypointsB):
    filteredMatches = []
    if len(matches) >= 4:
        src_pts = np.float32([ keypointsA[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ keypointsB[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 1.0)
        matchesMask = mask.ravel().tolist()

        for i in range(len(matchesMask)):
            if matchesMask[i] == 1:
                filteredMatches.append(matches[i])

    return homography, filteredMatches

def showResult(imgA, keypointsA, imgB, keypointsB, matches, homography=None):
    imgMatch = cv.drawMatches(imgA, keypointsA, imgB, keypointsB, matches, None)

    if homography is not None:
        h,w = imgA.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts, homography)
        dst += np.float32([w,0])
        imgMatch = cv.polylines(imgMatch,[np.int32(dst)],True,(0,255,0),3, cv.LINE_AA)

    cv.namedWindow("matches", cv.WINDOW_KEEPRATIO)
    cv.imshow("matches", imgMatch)
    cv.waitKey(0)

def main():
    ##################################################
    detector = cv.xfeatures2d.SIFT_create()
    #detector = cv.KAZE_create()
    matcher = cv.FlannBasedMatcher()
    ##################################################

    img = cv.imread('image1.png')
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #FAST
    # initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()
    # find and draw the keypoints
    kp_fast = fast.detect(grey,None)
    img2 = cv.drawKeypoints(img, kp_fast, None, color=(255,0,0))
    

    # SIFT
    # initiate SIFT object with default values
    sift = cv.xfeatures2d.SIFT_create()
    # find and draw the keypoints
    kp_sift = sift.detect(img,None)
    img3 = cv.drawKeypoints(grey,kp_sift,None, color=(255,0,0))


    # show results 
    cv.imshow('image FAST',img2)
    cv.imshow('image SIFT',img3)
    cv.waitKey(1000)

if __name__ == '__main__':
    main()
